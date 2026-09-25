import pytest
import torch
from torch.distributions import MultivariateNormal
from cheetah.particles import ParticleBeam
from gpsr.beams import NNTransform, NNParticleBeamGenerator, ResNNTransform
from gpsr.utils import to_linear_beam


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
    def test_resnn_transform_linear_forward(self, dim):
        transformer = ResNNTransform(n_hidden=2, width=10, phase_space_dim=dim)
        X = torch.rand(5, dim)

        output = transformer.linear_forward(X)

        assert output.shape == X.shape
        assert torch.is_tensor(output)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_resnn_transform_forward_without_skip_connection(self, dim):
        transformer = ResNNTransform(n_hidden=2, width=10, phase_space_dim=dim)
        X = torch.rand(5, dim)

        for layer in transformer.res_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        output = transformer(X)

        assert output.shape == X.shape
        assert torch.allclose(output, torch.zeros_like(X))

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_resnn_transform_forward_with_skip_connection(self, dim):
        transformer = ResNNTransform(
            n_hidden=2, width=10, phase_space_dim=dim, use_skip_connection=True
        )
        X = torch.rand(5, dim)

        with torch.no_grad():
            transformer.alpha.zero_()

        output = transformer(X)
        expected = transformer.linear_forward(X) * transformer.output_scale

        assert output.shape == X.shape
        assert torch.allclose(output, expected)

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

    def test_to_linear_beam_preserves_device_and_dtype(self):
        transformer = ResNNTransform(
            n_hidden=2,
            width=10,
            phase_space_dim=4,
            use_skip_connection=True,
        ).double()
        generator = NNParticleBeamGenerator(
            n_particles=10,
            energy=1e9,
            transformer=transformer,
            n_dim=4,
        )
        generator.base_particles = generator.base_particles.double()

        beam = to_linear_beam(generator)

        assert beam.particles.device == generator.base_particles.device
        assert beam.particles.dtype == generator.base_particles.dtype
