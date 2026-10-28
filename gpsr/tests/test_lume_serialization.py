"""Round-trip tests for the spec <-> model serialization in ``gpsr.lume``.

``build_gpsr_lume_model`` and ``serialize_gpsr_lume_model`` (see
``gpsr.lume.builders``) are inverses: serializing a built model gives back a spec
that rebuilds it, and re-serializing that gives the same spec again. A checkpoint
written by ``LitGPSRLUME`` must then reload through ``load_self_contained`` with
the trained weights and the calibrated lattice intact.

``build_accelerator`` below is the facility builder the test specs name by dotted
path, standing in for a real one. It is defined here rather than imported so the
tests need no facility package.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lume_cheetah")
lightning = pytest.importorskip("lightning")

import cheetah
from lume_cheetah import CheetahSimulator, LUMECheetahModel
from lume_cheetah.actions import (
    CheetahReadOnlyNDVariable,
    CheetahWritableScalarVariable,
)

from gpsr.beams import NNParticleBeamGenerator, Transform
from gpsr.lume.builders import (
    _lattice_json_to_segment,
    _segment_to_lattice_json,
    build_gpsr_lume_model,
    model_spec_from_files,
    serialize_gpsr_lume_model,
)
from gpsr.lume.model import GPSRLUMEModel
from gpsr.lume.training import LitGPSRLUME

ENERGY = 1.25e8
BUILDER_PATH = f"{__name__}.build_accelerator"


class QuadrupoleKVariable(CheetahWritableScalarVariable):
    """A quadrupole's geometric strength, with no unit conversion.

    The conversion is what a real facility variable adds; these tests only need
    a variable that reads and writes an element attribute.
    """

    unit: str = "1/m^2"
    element_attribute: str = "k1"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        return getattr(element, self.element_attribute)

    def _set(self, simulator, value):
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        setattr(element, self.element_attribute, torch.as_tensor(value))


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """A screen's image, transposed to the ``(x, y)`` order datasets use.

    Mirrors a facility image variable, minus the camera full-scale factor these
    tests do not need.
    """

    element_attribute: str = "reading"

    def _get(self, simulator):
        # `reading` is (..., y, x); `.mT` transposes only the last two axes.
        return super()._get(simulator).mT


def build_accelerator(lattice, name_map, energy: float) -> LUMECheetahModel:
    """Build the stand-in accelerator named by ``BUILDER_PATH``.

    Mirrors a facility builder's signature: ``lattice`` / ``name_map`` come from
    the spec's accelerator config, ``energy`` is passed separately by
    ``build_accelerator_from_spec``. Every screen in the lattice gets an image
    variable, keyed ``<ELEMENT>:IMAGE``.
    """
    segment = _lattice_json_to_segment(lattice)
    placeholder_beam = cheetah.ParticleBeam(
        torch.zeros(1, 7), energy=torch.tensor(energy)
    )
    action_variables = [
        QuadrupoleKVariable(name=control_name, element_name=element_name)
        for element_name, control_name in name_map.items()
    ]
    action_variables += [
        ScreenImageVariable(
            name=f"{element.name.upper()}:IMAGE",
            element_name=element.name,
            shape=tuple(element.resolution),
        )
        for element in segment.elements
        if isinstance(element, cheetah.Screen)
    ]
    return LUMECheetahModel(
        simulator=CheetahSimulator(
            segment=segment, initial_beam_distribution=placeholder_beam
        ),
        action_variables=action_variables,
    )


SCREEN_SHAPE = (16, 12)
SCREEN_PIXEL_SIZE = 1e-4
OBSERVATIONS_METADATA = {
    "SCR:IMAGE": {
        "type": "screen",
        "shape": SCREEN_SHAPE,
        "pixel_size": torch.tensor([SCREEN_PIXEL_SIZE, SCREEN_PIXEL_SIZE]),
    }
}


@pytest.fixture
def lattice_json(tmp_path):
    """Two quadrupoles, a drift and a screen, as a JSON file path.

    The screen is what makes this lattice representative: tracking through one
    leaves per-batch beam buffers on the element, which is the bulk of what
    ``on_save_checkpoint`` strips. The second quadrupole gives the device tests a
    magnet to hold fixed while the first is scanned.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Quadrupole(
                length=torch.tensor(0.1), k1=torch.tensor(1.0), name="q1"
            ),
            cheetah.Quadrupole(
                length=torch.tensor(0.1), k1=torch.tensor(1.0), name="q2"
            ),
            cheetah.Drift(length=torch.tensor(0.5), name="d1"),
            cheetah.Screen(
                resolution=SCREEN_SHAPE,
                pixel_size=torch.tensor([SCREEN_PIXEL_SIZE, SCREEN_PIXEL_SIZE]),
                is_active=True,
                method="cloud-in-cell",
                name="scr",
            ),
        ],
        name="test_segment",
    )
    path = tmp_path / "lattice.json"
    path.write_text(_segment_to_lattice_json(segment))
    return path


@pytest.fixture
def name_map_json(tmp_path):
    """Element-to-control name map for ``lattice_json``."""
    path = tmp_path / "name_map.json"
    path.write_text(json.dumps({"q1": "Q1:BCTRL", "q2": "Q2:BCTRL"}))
    return path


@pytest.fixture
def spec(lattice_json, name_map_json):
    """A spec built from files, as a caller would -- before any round-trip."""
    return model_spec_from_files(
        lattice_json,
        name_map_json,
        energy=ENERGY,
        accelerator_builder=BUILDER_PATH,
        generator={
            "cls": NNParticleBeamGenerator,
            "config": {"n_particles": 100, "n_dim": 6},
        },
    )


class TestSpecRoundTrip:
    def test_build_from_spec(self, spec):
        model = build_gpsr_lume_model(spec)

        assert isinstance(model, GPSRLUMEModel)
        assert float(model.beam_generator.energy) == pytest.approx(ENERGY)
        # The generator's energy is passed to the accelerator builder, so a spec
        # cannot express disagreement between the two.
        assert float(
            model.lume_cheetah_model.simulator.initial_beam_distribution.energy
        ) == pytest.approx(ENERGY)
        assert model.accelerator_spec is not None

    def test_serialize_is_inverse_of_build(self, spec):
        model = build_gpsr_lume_model(spec)
        reserialized = serialize_gpsr_lume_model(model)

        assert reserialized["generator"]["cls"] == (
            "gpsr.beams.NNParticleBeamGenerator"
        )
        assert reserialized["accelerator"]["builder"] == BUILDER_PATH
        assert reserialized["generator"]["config"]["n_particles"] == 100
        assert reserialized["generator"]["config"]["energy"] == pytest.approx(ENERGY)
        assert reserialized["accelerator"]["config"]["name_map"] == {
            "q1": "Q1:BCTRL",
            "q2": "Q2:BCTRL",
        }

    def test_round_trip_is_idempotent(self, spec):
        # Not textually equal to the file-read spec: the first pass rewrites the
        # lattice into cheetah's own JSON schema and fills in get_config()'s
        # defaults. So equality is asserted between the first and second passes.
        first = serialize_gpsr_lume_model(build_gpsr_lume_model(spec))
        second = serialize_gpsr_lume_model(build_gpsr_lume_model(first))

        assert second == first

    def test_spec_is_json_pure(self, spec):
        # The checkpoint constraint: everything must survive json, so that a
        # checkpoint loads with torch.load(weights_only=True).
        round_tripped = json.loads(json.dumps(spec))

        assert round_tripped == spec

    def test_serialize_captures_live_element_parameters(self, spec):
        # The lattice is re-read from the live segment, not copied from the
        # recorded spec, so a post-build calibration is captured.
        model = build_gpsr_lume_model(spec)
        segment = model.lume_cheetah_model.simulator.segment
        segment.q1.k1 = torch.tensor(7.5)

        reserialized = serialize_gpsr_lume_model(model)
        rebuilt = build_gpsr_lume_model(reserialized)

        assert float(
            rebuilt.lume_cheetah_model.simulator.segment.q1.k1
        ) == pytest.approx(7.5)

    def test_get_config_describes_the_model(self, spec):
        # The model describes itself the same way a beam generator does.
        model = build_gpsr_lume_model(spec)

        assert model.get_config() == serialize_gpsr_lume_model(model)

    def test_hand_assembled_model_cannot_describe_itself(self, spec):
        # No accelerator_spec means no record of how the accelerator was built,
        # and a built LUMECheetahModel does not say how it was built.
        built = build_gpsr_lume_model(spec)
        hand_assembled = GPSRLUMEModel(
            lume_cheetah_model=built.lume_cheetah_model,
            beam_generator=built.beam_generator,
        )

        with pytest.raises(NotImplementedError, match="accelerator_spec"):
            hand_assembled.get_config()

    def test_spec_without_an_accelerator_is_rejected(self, spec):
        with pytest.raises(KeyError, match="accelerator"):
            build_gpsr_lume_model({"generator": spec["generator"]})


class TestCheckpointRoundTrip:
    def test_load_self_contained_restores_weights_and_lattice(self, spec, tmp_path):
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        # Stand in for training: perturb the trained weights and the calibration
        # so a successful reload has something to prove.
        with torch.no_grad():
            for parameter in lit.gpsr_lume_model.beam_generator.parameters():
                parameter.add_(0.1)
        # Calibrate the drift, not the quadrupole: the forward pass below writes
        # Q1:BCTRL, so a quadrupole calibration would just be overwritten by it.
        lit.gpsr_lume_model.lume_cheetah_model.simulator.segment.d1.length = (
            torch.tensor(0.75)
        )
        expected = {
            key: value.clone()
            for key, value in lit.gpsr_lume_model.beam_generator.state_dict().items()
        }

        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        _forward_pass(lit)
        trainer.save_checkpoint(checkpoint_path)

        reloaded = LitGPSRLUME.load_self_contained(checkpoint_path, map_location="cpu")

        restored = reloaded.gpsr_lume_model.beam_generator.state_dict()
        assert set(restored) == set(expected)
        for key, value in expected.items():
            assert torch.equal(restored[key], value), key
        assert float(
            reloaded.gpsr_lume_model.lume_cheetah_model.simulator.segment.d1.length
        ) == pytest.approx(0.75)
        # The reloaded model runs, so the stripped subtree was restored intact.
        _forward_pass(reloaded)

    def test_checkpoint_loads_with_weights_only(self, spec, tmp_path):
        # The JSON-purity constraint, end to end: no pickled class or function
        # anywhere in the payload. Saved after a forward pass, since that is when
        # the accelerator holds tensors that are not plain parameters.
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        _forward_pass(lit)
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True)

        assert LitGPSRLUME._SPEC_KEY in checkpoint
        assert json.loads(json.dumps(checkpoint[LitGPSRLUME._SPEC_KEY]))

    def test_missing_generator_key_raises(self, spec, tmp_path):
        # strict_loading=True must still catch a genuinely mismatched generator
        # (a missing/unexpected beam_generator key), even though on_load_checkpoint
        # backfills the frozen accelerator subtree before this check runs.
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        del checkpoint["state_dict"]["gpsr_lume_model.beam_generator.particle_charges"]
        torch.save(checkpoint, checkpoint_path)

        fresh_model = build_gpsr_lume_model(spec)
        with pytest.raises(RuntimeError, match="Missing key"):
            LitGPSRLUME.load_from_checkpoint(
                checkpoint_path, gpsr_lume_model=fresh_model, map_location="cpu"
            )

    def test_frozen_accelerator_is_stripped_from_state_dict(self, spec, tmp_path):
        # Saved after a forward pass, as training does: tracking leaves per-batch
        # beam buffers on the screen that a freshly built model does not have, and
        # those are the bulk of what the strip removes.
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        _forward_pass(lit)
        assert [key for key in lit.state_dict() if "_read_beam" in key], (
            "the forward pass left no per-batch buffers, so this test would pass "
            "even without the strip"
        )

        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True)

        assert not [
            key
            for key in checkpoint["state_dict"]
            if key.startswith(LitGPSRLUME._LUME_PREFIX)
        ]
        assert [
            key
            for key in checkpoint["state_dict"]
            if key.startswith("gpsr_lume_model.beam_generator.")
        ]


class TestConstantsDevice:
    """Beamline constants reach the lattice wherever the caller left them.

    They come from ``source_info``, not the batch, so Lightning does not move
    them with everything else.
    """

    def test_lattice_device_reads_the_segment(self, spec):
        model = build_gpsr_lume_model(spec)

        assert model._lattice_device() == (
            model.lume_cheetah_model.simulator.segment.q1.k1.device
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "constant", [torch.tensor(5.0), torch.tensor([5.0]), 5.0], ids=type
    )
    def test_a_cpu_constant_tracks_on_a_cuda_model(self, spec, constant):
        model = build_gpsr_lume_model(spec).to("cuda")

        images = model(
            settings={"Q1:BCTRL": torch.tensor([1.0, 2.0], device="cuda")},
            observations_metadata=OBSERVATIONS_METADATA,
            beamline_constants={"Q2:BCTRL": constant},
        )

        assert images["SCR:IMAGE"].shape == (2, *SCREEN_SHAPE)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_the_constant_reaches_the_element(self, spec):
        # The move has to land on the element, not just survive the arithmetic:
        # this variable assigns k1 straight through with no unit conversion.
        model = build_gpsr_lume_model(spec).to("cuda")

        model(
            settings={"Q1:BCTRL": torch.tensor([1.0, 2.0], device="cuda")},
            observations_metadata=OBSERVATIONS_METADATA,
            beamline_constants={"Q2:BCTRL": torch.tensor(5.0)},
        )

        q2 = model.lume_cheetah_model.simulator.segment.q2
        assert q2.k1.device.type == "cuda"
        assert float(q2.k1) == pytest.approx(5.0)


class TestGeneratorConfig:
    def test_custom_transformer_survives_a_checkpoint(self, spec, tmp_path):
        # The spec records the transformer's architecture, so a non-default one is
        # rebuilt by load_self_contained rather than replaced by the default.
        spec["generator"]["config"]["transformer"] = {
            "cls": "gpsr.beams.NNTransform",
            "config": {
                "n_hidden": 4,
                "width": 64,
                "activation": "torch.nn.SiLU",
                "output_scale": 7.5e-3,
                "phase_space_dim": 6,
            },
        }
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        with torch.no_grad():
            for parameter in lit.gpsr_lume_model.beam_generator.parameters():
                parameter.add_(0.1)
        expected = {
            key: value.clone()
            for key, value in lit.gpsr_lume_model.beam_generator.state_dict().items()
        }

        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        trainer.save_checkpoint(checkpoint_path)

        reloaded = LitGPSRLUME.load_self_contained(checkpoint_path, map_location="cpu")

        transformer = reloaded.gpsr_lume_model.beam_generator.transformer
        assert isinstance(transformer.stack[1], torch.nn.SiLU)
        assert float(transformer.output_scale) == pytest.approx(7.5e-3)
        restored = reloaded.gpsr_lume_model.beam_generator.state_dict()
        assert set(restored) == set(expected)
        for key, value in expected.items():
            assert torch.equal(restored[key], value), key

    def test_generator_config_in_a_checkpoint_is_json_pure(self, spec, tmp_path):
        # The nested transformer entry must not break the weights_only=True load.
        lit = LitGPSRLUME(build_gpsr_lume_model(spec), lr=1e-3)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        generator = checkpoint[LitGPSRLUME._SPEC_KEY]["generator"]

        assert json.loads(json.dumps(generator)) == generator
        assert generator["config"]["transformer"]["cls"] == "gpsr.beams.NNTransform"

    def test_checkpoint_with_an_undescribable_transformer_has_no_spec(
        self, spec, tmp_path
    ):
        # A transformer whose get_config raises propagates through the generator,
        # and is handled the same way as a generator that cannot describe itself.
        model = build_gpsr_lume_model(spec)
        model.beam_generator.transformer = _TransformWithoutConfig()
        lit = LitGPSRLUME(model, lr=1e-3)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)

        with pytest.warns(UserWarning, match="self-contained spec"):
            trainer.save_checkpoint(checkpoint_path)

        assert LitGPSRLUME._SPEC_KEY not in torch.load(
            checkpoint_path, weights_only=True
        )

    def test_checkpoint_without_a_generator_config_has_no_spec(self, spec, tmp_path):
        # A generator whose get_config raises is skipped with a warning: the
        # checkpoint stays valid, but load_self_contained cannot use it.
        model = build_gpsr_lume_model(spec)
        model.beam_generator = _GeneratorWithoutConfig(n_particles=100, energy=ENERGY)
        lit = LitGPSRLUME(model, lr=1e-3)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer = _trainer()
        trainer.strategy.connect(lit)

        with pytest.warns(UserWarning, match="self-contained spec"):
            trainer.save_checkpoint(checkpoint_path)

        assert LitGPSRLUME._SPEC_KEY not in torch.load(
            checkpoint_path, weights_only=True
        )
        with pytest.raises(KeyError, match=LitGPSRLUME._SPEC_KEY):
            LitGPSRLUME.load_self_contained(checkpoint_path, map_location="cpu")


class _GeneratorWithoutConfig(NNParticleBeamGenerator):
    """A generator whose ``get_config`` raises, as an external one's may."""

    def get_config(self) -> dict:
        raise NotImplementedError("test generator has no config")


class _TransformWithoutConfig(Transform):
    """A transform that leaves ``get_config`` raising, as an external one may."""

    def forward(self, X):
        return X


def _trainer():
    """A Trainer that neither logs nor checkpoints on its own."""
    return lightning.Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False, max_epochs=1
    )


def _forward_pass(lit: LitGPSRLUME) -> None:
    """Track a two-step scan and read the screen, as training does every step."""
    lit.gpsr_lume_model(
        settings={"Q1:BCTRL": torch.tensor([1.0, 2.0])},
        observations_metadata=OBSERVATIONS_METADATA,
    )
