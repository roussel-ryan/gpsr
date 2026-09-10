import torch
from torch import Tensor
from tensordict import TensorDict

from cheetah.particles import ParticleBeam
from gpsr.beams import BeamGenerator

from lume_cheetah import LUMECheetahModel


class GPSRLUMEModel(torch.nn.Module):
    """A beam generator paired with a frozen LUME-Cheetah virtual accelerator.

    ``beam_generator`` samples a beam at the reconstruction point (the only
    trainable piece); ``lume_cheetah_model`` tracks it through the lattice to
    produce observable PVs. Call the model (see :meth:`forward`) with beamline
    settings and observation metadata to predict images; use
    :meth:`predict_multi_source` for a multi-source batch with a single shared
    beam sample.

    Parameters
    ----------
    lume_cheetah_model : LUMECheetahModel
        The frozen virtual accelerator.
    beam_generator : BeamGenerator
        The trainable beam generator. Must share ``lume_cheetah_model``'s reference
        energy (checked below).
    accelerator_spec : dict, optional
        ``{"builder": <import path>, "config": {...}}`` -- the recipe that produced
        ``lume_cheetah_model``, recorded as build provenance so
        :func:`gpsr.lume.builders.serialize_gpsr_lume_model` can round-trip it into a
        self-contained checkpoint. Set by
        :func:`gpsr.lume.builders.build_gpsr_lume_model`; ``None`` for a
        hand-assembled accelerator, which then cannot be serialized. A
        ``LUMECheetahModel`` is a flat collection of per-PV action variables with no
        facility-wide mapping object to invert, so the recipe cannot be recovered
        from the built model.
    """

    def __init__(
        self,
        lume_cheetah_model: LUMECheetahModel,
        beam_generator: BeamGenerator,
        accelerator_spec: dict | None = None,
    ):
        super().__init__()

        # The accelerator converts EPICS magnet settings (BCTRL/BDES) to Cheetah
        # geometric strengths via magnetic rigidity, which depends on beam energy.
        # That energy is frozen into `simulator.energies` at construction from the
        # initial (placeholder) beam, while the beam actually tracked is the
        # generator's. The two MUST share one reference energy, or every magnet
        # setting maps to the wrong strength. The spec build path takes the
        # accelerator's energy *from* the generator, so it cannot disagree; this
        # guards the direct-injection path, where the two arrive independently.
        generator_energy = beam_generator.energy
        accelerator_energy = (
            lume_cheetah_model.simulator.initial_beam_distribution.energy
        )
        if not torch.allclose(generator_energy, accelerator_energy):
            raise ValueError(
                f"Beam-generator energy ({float(generator_energy):.6g} eV) does not "
                f"match the LUME-Cheetah accelerator energy "
                f"({float(accelerator_energy):.6g} eV). They must share one reference "
                f"energy: the accelerator's magnetic-rigidity conversion uses its own "
                f"energy while tracking the generator's beam."
            )

        self.lume_cheetah_model = lume_cheetah_model
        self.beam_generator = beam_generator
        # Plain dict, deliberately not a buffer/parameter: it is JSON-pure build
        # provenance for the checkpoint spec, not model state.
        self.accelerator_spec = accelerator_spec

    def forward(
        self,
        settings: dict[str, Tensor] | TensorDict,
        observations_metadata: dict,
        beam: ParticleBeam | None = None,
    ) -> dict[str, Tensor]:
        """Apply beamline ``settings``, track the beam, and return the observable
        PVs described by ``observations_metadata`` (e.g. screen images).

        Parameters
        ----------
        settings : dict[str, Tensor] | TensorDict
            Beamline settings (e.g. magnet BCTRL/BDES values) to apply before
            tracking.
        observations_metadata : dict
            Per-observable metadata (see :meth:`_setup_observable_elements`).
            Keys are the observation PVs to return; the metadata is applied on
            every call so screen configuration is explicit rather than implicit
            state the caller must remember to set beforehand.
        beam : ParticleBeam | None, default=None
            Reconstruction-point beam to track. If ``None``, samples one from
            ``self.beam_generator()``. Pass the *same* beam across multiple calls
            (e.g. one per source in a multi-source batch) for a joint
            reconstruction against a shared sample; let it default per call for
            independent samples.
        """
        if beam is None:
            beam = self.beam_generator()
        self.lume_cheetah_model.simulator.beam_distribution = beam
        self._setup_observable_elements(observations_metadata)
        self.lume_cheetah_model.set(settings)
        return self.lume_cheetah_model.get(list(observations_metadata.keys()))

    def predict_multi_source(
        self,
        batch: dict[str, TensorDict],
        source_info: dict[str, dict],
    ) -> dict[str, dict[str, Tensor]]:
        """Predict observations for every source in one multi-source batch.

        Samples a single beam at the reconstruction point and evaluates every
        source against that *same* beam (each with its own observation metadata
        and beamline constants) so the reconstruction is jointly consistent
        across sources. This is the entry point trainers should use for
        multi-source batches; :meth:`forward` handles the single-source case.

        Parameters
        ----------
        batch : dict[str, TensorDict]
            ``batch[source_name]`` -> TensorDict with keys ``"beamline_settings"``,
            ``"observations"``.
        source_info : dict[str, dict]
            Per-source ``observations_metadata`` + ``beamline_constants`` (as
            exposed by ``GPSRLUMEDataModule.source_info``). Must cover every
            source in ``batch``.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            Per source: predicted images keyed by observation PV.
        """
        beam = self.beam_generator()  # one sample shared across all sources
        return {
            source_name: self(
                settings=dict(subbatch["beamline_settings"])
                | source_info[source_name]["beamline_constants"],
                observations_metadata=source_info[source_name]["observations_metadata"],
                beam=beam,
            )
            for source_name, subbatch in batch.items()
        }

    def _setup_observable_elements(
        self,
        observations_metadata: dict,
    ):
        """Configure observation elements based on metadata.

        Invoked from :meth:`forward` on every prediction call.

        Parameters
        ----------
        observations_metadata : dict
            Dictionary mapping observation PV names to their metadata
            configuration, where metadata should contain 'type'.
            If 'type' is 'screen', metadata should also contain 'shape' and
            'pixel_size' keys.

        Note
        ----
        Only observation type 'screen' is supported as of now.
        """
        for observation_pv_name, metadata in observations_metadata.items():
            if metadata["type"] == "screen":
                self._setup_screen(
                    observation_pv_name=observation_pv_name,
                    resolution=metadata["shape"],
                    pixel_size=metadata["pixel_size"],
                )
            else:
                raise NotImplementedError(
                    f"Unsupported observation type {metadata['type']!r} in metadata: {metadata}"
                )

    def _setup_screen(
        self,
        observation_pv_name: str,
        resolution: tuple[int, int],
        pixel_size: Tensor,
    ):
        """Configure a screen element for observation.

        Invoked from :meth:`forward` via :meth:`_setup_observable_elements` on
        every prediction call.

        Parameters
        ----------
        observation_pv_name : str
            The image PV name, i.e. this observation's key in
            ``observations_metadata``. Looked up in the LUME model's
            ``supported_variables`` to reach the Cheetah element behind it.
        resolution : tuple[int, int]
            The resolution (width, height) of the screen detector.
        pixel_size : Tensor
            The physical pixel size of the detector.
        """
        # The observation's own key is the PV to look up -- no facility-specific PV
        # naming convention is assumed here. :meth:`forward` already fetches these
        # keys straight from the LUME model, so a key that is not a supported
        # variable could not be read back anyway.
        try:
            image_variable = self.lume_cheetah_model.supported_variables[
                observation_pv_name
            ]
        except KeyError:
            raise KeyError(
                f"Observation {observation_pv_name!r} is not a variable of this LUME "
                f"model, so there is no screen behind it to configure. The keys of "
                f"'observations_metadata' must be PVs the model supports; it "
                f"supports {sorted(self.lume_cheetah_model.supported_variables)}."
            ) from None
        # The image PV's action variable carries the name of the Cheetah element it
        # reads, so the PV -> element lookup needs no facility-specific mapping
        # object: `element_name` is declared on lume_cheetah's generic action base.
        screen_lattice_name = image_variable.element_name

        screen_element = getattr(
            self.lume_cheetah_model.simulator.segment, screen_lattice_name
        )
        screen_element.resolution = resolution
        screen_element.is_active = True
        pixel_size = pixel_size.to(
            screen_element.pixel_size.device
        )  # match the segment's device
        screen_element.pixel_size = pixel_size
        # One imaging method, always: `cloud-in-cell` is differentiable, so the same
        # image the fit takes gradients through is the one prediction reports. Set here
        # rather than trusted from the lattice, since a `histogram` screen would make
        # the model silently untrainable.
        screen_element.method = "cloud-in-cell"

        # set resolution in LumeModel PV. `supported_variables` returns a fresh dict
        # each call but the Variable objects themselves are shared, so mutating the
        # shape here does reach the model's registry.
        image_variable.shape = tuple(resolution)
        self.lume_cheetah_model.update_state()
