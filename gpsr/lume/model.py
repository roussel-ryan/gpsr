import torch
from torch import Tensor
from tensordict import TensorDict

from cheetah.particles import ParticleBeam
from gpsr.beams import BeamGenerator

from lume_cheetah import LUMECheetahModel


def _merge_settings(
    beamline_settings: dict[str, Tensor] | TensorDict,
    beamline_constants: dict[str, Tensor] | None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """Combine scanned settings with fixed constants, rejecting shared keys.

    A key in both would take its constant value, silently replacing the scanned
    one and leaving the scan looking flat.

    Constants are moved to ``device``, and may be given on any device or as plain
    numbers. They arrive from ``source_info`` rather than through the batch, so
    Lightning never sees them and leaves them behind on the CPU. Settings are the
    caller's to place.
    """
    settings = dict(beamline_settings)
    # Testing None rather than falsiness: bool() on a TensorDict raises.
    constants = dict(beamline_constants) if beamline_constants is not None else {}

    shared = settings.keys() & constants.keys()
    if shared:
        raise ValueError(
            f"beamline_settings and beamline_constants share the key(s) "
            f"{sorted(shared)}; a parameter is either scanned or held fixed, "
            f"not both."
        )
    if device is not None:
        # `as_tensor` first, so a plain number works as a constant.
        constants = {
            key: torch.as_tensor(value).to(device) for key, value in constants.items()
        }
    return settings | constants


class GPSRLUMEModel(torch.nn.Module):
    """A beam generator paired with a frozen LUME-Cheetah virtual accelerator.

    ``beam_generator`` samples a beam at the reconstruction point (the only
    trainable piece); ``lume_cheetah_model`` tracks it through the lattice to
    produce observable PVs. Call the model (see ``forward``) with beamline
    settings and observation metadata to predict images; use
    ``predict_multi_source`` for a multi-source batch with a single shared
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
        ``lume_cheetah_model``, which ``get_config`` needs. Set by
        ``build_gpsr_lume_model``; ``None`` for a hand-assembled accelerator, which
        then cannot describe itself, since a built ``LUMECheetahModel`` does not say
        how it was built.
    """

    def __init__(
        self,
        lume_cheetah_model: LUMECheetahModel,
        beam_generator: BeamGenerator,
        accelerator_spec: dict | None = None,
    ):
        super().__init__()

        # The accelerator converts EPICS magnet settings to Cheetah geometric
        # strengths via magnetic rigidity, which depends on the energy frozen into
        # `simulator.energies` at construction -- while the beam actually tracked is
        # the generator's. The two must share one reference energy or every magnet
        # setting maps to the wrong strength. The spec build path threads one value
        # into both; this guards direct injection, where they arrive independently.
        generator_energy = getattr(beam_generator, "energy", None)
        if generator_energy is None:
            raise TypeError(
                f"{type(beam_generator).__name__} has no 'energy' attribute. A "
                f"generator used with GPSRLUMEModel must expose its reference "
                f"energy [eV] as a buffer, attribute or property, so that it can be "
                f"matched against the accelerator's."
            )
        accelerator_energy = (
            lume_cheetah_model.simulator.initial_beam_distribution.energy
        )
        generator_energy_tensor = torch.as_tensor(
            generator_energy,
            dtype=accelerator_energy.dtype,
            device=accelerator_energy.device,
        )
        if not torch.allclose(generator_energy_tensor, accelerator_energy):
            raise ValueError(
                f"Beam-generator energy ({float(generator_energy):.6g} eV) does not "
                f"match the LUME-Cheetah accelerator energy "
                f"({float(accelerator_energy):.6g} eV). They must share one reference "
                f"energy: the accelerator's magnetic-rigidity conversion uses its own "
                f"energy while tracking the generator's beam."
            )

        # Cheetah elements accept an nn.Parameter for any attribute, which Adam
        # would then train; freeze explicitly rather than rely on builders never
        # passing one.
        lume_cheetah_model.requires_grad_(False)
        self.lume_cheetah_model = lume_cheetah_model
        self.beam_generator = beam_generator
        # A plain dict, not a buffer or parameter: it records how the accelerator was
        # built, which is not part of the model's state.
        self.accelerator_spec = accelerator_spec

    def get_config(self) -> dict:
        """Return JSON-serializable kwargs that rebuild this model.

        Same contract as ``BeamGenerator.get_config``: pass the result to
        ``build_gpsr_lume_model`` to get an equivalent but *untrained* model.
        Trained weights are not included -- they are saved and restored separately
        through the ``state_dict``.

        Raises
        ------
        NotImplementedError
            If the beam generator cannot describe itself, or if the accelerator was
            assembled by hand and so carries no recipe. Such a model must be rebuilt
            and re-supplied explicitly rather than loaded from a checkpoint alone.
        """
        # Imported here because `builders` imports this module.
        from gpsr.lume.builders import serialize_gpsr_lume_model

        return serialize_gpsr_lume_model(self)

    def forward(
        self,
        settings: dict[str, Tensor] | TensorDict,
        observations_metadata: dict,
        beam: ParticleBeam | None = None,
        beamline_constants: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Apply beamline ``settings``, track the beam, and return the observable
        PVs described by ``observations_metadata`` (e.g. screen images).

        Parameters
        ----------
        settings : dict[str, Tensor] | TensorDict
            Scanned beamline settings (e.g. magnet BCTRL/BDES values), one value
            per scan step.
        observations_metadata : dict
            Per-observable metadata (see ``_setup_observable_elements``). Keys are
            the observation PVs to return; it is applied on every call, so screen
            configuration is never leftover state from a previous one.
        beam : ParticleBeam | None, default=None
            Reconstruction-point beam to track. If ``None``, samples one from
            ``self.beam_generator()``. Pass the *same* beam across multiple calls
            (e.g. one per source in a multi-source batch) for a joint
            reconstruction against a shared sample; let it default per call for
            independent samples.
        beamline_constants : dict[str, Tensor] | None, default=None
            Parameters held fixed for the whole scan, applied alongside
            ``settings``. Its keys must not appear in ``settings``. Moved to the
            lattice's device, so they may be given on any device; ``settings``
            must already be there.

        Raises
        ------
        ValueError
            If a key appears in both ``settings`` and ``beamline_constants``.
        """
        if beam is None:
            beam = self.beam_generator()
        self.lume_cheetah_model.simulator.beam_distribution = beam
        self._setup_observable_elements(observations_metadata)
        self.lume_cheetah_model.set(
            _merge_settings(settings, beamline_constants, self._lattice_device())
        )
        return self.lume_cheetah_model.get(list(observations_metadata.keys()))

    def _lattice_device(self) -> torch.device | None:
        """The device the Cheetah lattice is on, or ``None`` if it holds no tensors.

        Read from the lattice rather than from the beam, which may be supplied by
        a caller and sit elsewhere.
        """
        segment = self.lume_cheetah_model.simulator.segment
        return next((buffer.device for buffer in segment.buffers()), None)

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
        multi-source batches; ``forward`` handles the single-source case.

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
                settings=subbatch["beamline_settings"],
                observations_metadata=source_info[source_name]["observations_metadata"],
                beam=beam,
                beamline_constants=source_info[source_name]["beamline_constants"],
            )
            for source_name, subbatch in batch.items()
        }

    def _setup_observable_elements(
        self,
        observations_metadata: dict,
    ):
        """Configure observation elements based on metadata.

        Invoked from ``forward`` on every prediction call.

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

        Invoked from ``forward`` via ``_setup_observable_elements`` on
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
        # The observation's own key is the PV to look up, so no facility-specific
        # naming convention is assumed here.
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
        # `element_name` is declared on lume_cheetah's generic action base, so the
        # PV -> element lookup needs no facility-specific mapping object.
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
        # Always `cloud-in-cell`, which is differentiable. Forced here rather than
        # trusted from the lattice, since a `histogram` screen would make the model
        # silently untrainable.
        screen_element.method = "cloud-in-cell"

        # `supported_variables` returns a fresh dict each call, but the Variable
        # objects are shared, so mutating the shape reaches the model's registry.
        image_variable.shape = tuple(resolution)
        self.lume_cheetah_model.update_state()
