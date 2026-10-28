import json

import pytest
import torch
from torch.distributions import MultivariateNormal
from cheetah.particles import ParticleBeam
from gpsr.beams import NNTransform, NNParticleBeamGenerator, Transform


class TestBeams:
    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_transform_initialization(self, dim):
        # Test NNTransform initialization
        n_hidden = 2
        width = 20
        dropout = 0.1
        output_scale = 1e-2
        activation = torch.nn.Tanh()

        transformer = NNTransform(
            n_hidden=n_hidden,
            width=width,
            dropout=dropout,
            activation=activation,
            output_scale=output_scale,
            phase_space_dim=dim,
        )

        assert isinstance(transformer.stack, torch.nn.Sequential)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_transform_forward(self, dim):
        # Test forward pass for NNTransform
        transformer = NNTransform(2, 10, phase_space_dim=dim)
        X = torch.rand(5, dim)

        output = transformer(X)
        assert output.shape == X.shape
        assert torch.is_tensor(output)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_particle_beam_generator_initialization(self, dim):
        # Test NNParticleBeamGenerator initialization
        n_particles = 1000
        energy = 1e9  # Beam energy in eV
        base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        transformer = NNTransform(2, 20)

        generator = NNParticleBeamGenerator(
            n_particles=n_particles,
            energy=energy,
            base_dist=base_dist,
            transformer=transformer,
        )

        assert generator.beam_energy.item() == energy
        assert generator.base_particles.shape == (n_particles, dim)
        assert isinstance(generator.transformer, NNTransform)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_particle_beam_generator_set_base_particles(self, dim):
        # Test set_base_particles method
        n_particles = 1000
        energy = 1e9
        generator = NNParticleBeamGenerator(n_particles, energy, n_dim=dim)

        generator.set_base_particles(500)
        assert generator.base_particles.shape == (500, dim)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_particle_beam_generator_forward(self, dim):
        # Initialize generator
        n_particles = 500
        energy = 1e9
        generator = NNParticleBeamGenerator(n_particles, energy, n_dim=dim)

        beam = generator.forward()

        # Assert the output of the forward method is an instance of ParticleBeam
        assert isinstance(beam, ParticleBeam)

        # check to make sure emittances are not nan
        assert not torch.isnan(beam.emittance_x)
        assert not torch.isnan(beam.emittance_y)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_transform_get_config_round_trips(self, dim):
        transformer = NNTransform(
            n_hidden=3,
            width=32,
            dropout=0.1,
            activation=torch.nn.SiLU(),
            output_scale=7.5e-3,
            phase_space_dim=dim,
        )

        rebuilt = NNTransform.from_config(transformer.get_config())
        rebuilt.load_state_dict(transformer.state_dict())

        transformer.eval()
        rebuilt.eval()
        X = torch.rand(5, dim)
        with torch.no_grad():
            assert torch.allclose(transformer(X), rebuilt(X))

    def test_nn_transform_accepts_an_activation_import_path(self):
        transformer = NNTransform(2, 20, activation="torch.nn.SiLU")

        assert isinstance(transformer.stack[1], torch.nn.SiLU)


class TestGeneratorConfig:
    """``get_config`` / ``from_config``, which back self-contained checkpoints."""

    def test_default_generator_round_trips(self):
        generator = NNParticleBeamGenerator(n_particles=100, energy=1.25e8)

        rebuilt = NNParticleBeamGenerator.from_config(generator.get_config())
        rebuilt.load_state_dict(generator.state_dict())

        generator.eval()
        rebuilt.eval()
        with torch.no_grad():
            assert torch.allclose(
                generator.transformer(generator.base_particles),
                rebuilt.transformer(rebuilt.base_particles),
            )

    def test_custom_transformer_round_trips(self):
        # The transformer's own architecture is recorded, so a non-default one
        # rebuilds and its weights load strictly.
        generator = NNParticleBeamGenerator(
            n_particles=100,
            energy=1.25e8,
            n_dim=4,
            transformer=NNTransform(
                4,
                64,
                dropout=0.1,
                activation=torch.nn.SiLU(),
                output_scale=7.5e-3,
                phase_space_dim=4,
            ),
        )

        rebuilt = NNParticleBeamGenerator.from_config(generator.get_config())
        rebuilt.load_state_dict(generator.state_dict())

        generator.eval()
        rebuilt.eval()
        with torch.no_grad():
            assert torch.allclose(
                generator.transformer(generator.base_particles),
                rebuilt.transformer(rebuilt.base_particles),
            )

    def test_config_is_json_pure(self):
        generator = NNParticleBeamGenerator(
            n_particles=100, energy=1.25e8, transformer=NNTransform(4, 64)
        )
        config = generator.get_config()

        assert json.loads(json.dumps(config)) == config

    def test_transformer_can_be_given_as_a_config_dict(self):
        generator = NNParticleBeamGenerator(
            n_particles=100,
            energy=1.25e8,
            transformer={
                "cls": "gpsr.beams.NNTransform",
                "config": {"n_hidden": 3, "width": 32},
            },
        )

        assert isinstance(generator.transformer, NNTransform)
        assert generator.transformer.stack[0].out_features == 32

    @pytest.mark.parametrize(
        "activation",
        [torch.nn.LeakyReLU(0.2), torch.nn.ELU(alpha=0.5), torch.nn.PReLU()],
    )
    def test_activation_that_cannot_be_rebuilt_is_rejected(self, activation):
        # Only the activation's class is recorded, so one carrying constructor
        # arguments or learnable parameters must raise instead of round-tripping
        # into a different function.
        transformer = NNTransform(2, 20, activation=activation)

        with pytest.raises(NotImplementedError, match="ctivation"):
            transformer.get_config()

    @pytest.mark.parametrize(
        "activation", [torch.nn.Tanh(), torch.nn.SiLU(), torch.nn.LeakyReLU()]
    )
    def test_activation_with_default_arguments_is_accepted(self, activation):
        transformer = NNTransform(2, 20, activation=activation)

        rebuilt = NNTransform.from_config(transformer.get_config())

        assert type(rebuilt.stack[1]) is type(activation)

    def test_custom_base_dist_is_rejected(self):
        generator = NNParticleBeamGenerator(
            n_particles=100,
            energy=1.25e8,
            base_dist=MultivariateNormal(torch.zeros(6), 2.0 * torch.eye(6)),
        )

        with pytest.raises(NotImplementedError, match="base_dist"):
            generator.get_config()

    def test_transformer_without_a_config_is_rejected(self):
        generator = NNParticleBeamGenerator(
            n_particles=100, energy=1.25e8, transformer=_TransformWithoutConfig()
        )

        with pytest.raises(NotImplementedError, match="get_config"):
            generator.get_config()


class _TransformWithoutConfig(Transform):
    """A transform that leaves ``get_config`` raising, as an external one may."""

    def forward(self, X):
        return X
