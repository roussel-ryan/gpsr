import torch
from torch.distributions import MultivariateNormal
from cheetah.particles import ParticleBeam
from gpsr.beams import NNTransform, NNParticleBeamGenerator


class TestBeams:
    def test_nn_transform_initialization(self):
        # Test NNTransform initialization
        n_hidden = 2
        width = 20
        dropout = 0.1
        output_scale = 1e-2
        phase_space_dim = 6
        activation = torch.nn.Tanh()

        transformer = NNTransform(
            n_hidden=n_hidden,
            width=width,
            dropout=dropout,
            activation=activation,
            output_scale=output_scale,
            phase_space_dim=phase_space_dim,
        )

        assert isinstance(transformer.stack, torch.nn.Sequential)

    def test_nn_transform_forward(self):
        # Test forward pass for NNTransform
        phase_space_dim = 6
        transformer = NNTransform(2, 10, phase_space_dim=phase_space_dim)
        X = torch.rand(5, phase_space_dim)

        output = transformer(X)
        assert output.shape == X.shape
        assert torch.is_tensor(output)

    def test_nn_particle_beam_generator_initialization(self):
        # Test NNParticleBeamGenerator initialization
        n_particles = 1000
        energy = 1e9  # Beam energy in eV
        base_dist = MultivariateNormal(torch.zeros(6), torch.eye(6))
        transformer = NNTransform(2, 20)

        generator = NNParticleBeamGenerator(
            n_particles=n_particles,
            energy=energy,
            base_dist=base_dist,
            transformer=transformer,
        )

        assert generator.beam_energy.item() == energy
        assert generator.base_particles.shape == (n_particles, 6)
        assert isinstance(generator.transformer, NNTransform)

    def test_nn_particle_beam_generator_set_base_particles(self):
        # Test set_base_particles method
        n_particles = 1000
        energy = 1e9
        generator = NNParticleBeamGenerator(n_particles, energy)

        generator.set_base_particles(500)
        assert generator.base_particles.shape == (500, 6)

    def test_nn_particle_beam_generator_forward(self):
        # Initialize generator
        n_particles = 500
        energy = 1e9
        generator = NNParticleBeamGenerator(n_particles, energy)

        beam = generator.forward()

        # Assert the output of the forward method is an instance of ParticleBeam
        assert isinstance(beam, ParticleBeam)
