import pytest
import torch
from torch.distributions import MultivariateNormal
from cheetah.particles import ParticleBeam
from gpsr.beams import NNTransform, NNParticleBeamGenerator
from gpsr.beams import compute_batched_jacobian


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
    def test_jacobian_sin(self, dim):
        # Compute Jacobian of f(x) = sin(x). The Jacobian matrix
        # is J = diag(cos(x1), ..., cos(xn)).
        def transform(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(x)

        def transform_deriv(x: torch.Tensor) -> torch.Tensor:
            return torch.cos(x)

        batch_size = 100
        x = torch.randn(batch_size, dim)
        J = compute_batched_jacobian(x, transform)

        # Test correct shape of Jacobian matrix.
        assert J.shape == (x.shape[0], x.shape[1], x.shape[1])

        # Test correct values of Jacobian matrix.
        xp = transform_deriv(x)
        for i in range(x.shape[0]):
            correct = torch.diag(torch.cos(x[i]))
            assert torch.all(torch.isclose(J[i], correct))

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_nn_transform_jacobian(self, dim):
        # Test batched Jacobian matrix calculation for NNTransform.
        transformer = NNTransform(2, 10, phase_space_dim=dim)
        x = torch.rand(5, dim)
        jacobians = compute_batched_jacobian(x, transformer)
        for i, jacobian in enumerate(jacobians):
            jacobian_exp = torch.func.jacrev(transformer)(x[i, :])
            assert torch.all(torch.isclose(jacobian, jacobian_exp))

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
