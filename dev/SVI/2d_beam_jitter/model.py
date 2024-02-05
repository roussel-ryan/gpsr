import matplotlib.pyplot as plt
import pyro
import torch
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule

from bmadx.bmad_torch.track_torch import Beam
from phase_space_reconstruction.histogram import histogram2d
from phase_space_reconstruction.modeling import NNTransform, InitialBeam


class BayesianOffsetBeam(PyroModule):
    def __init__(self, base_beam):
        super(BayesianOffsetBeam, self).__init__()
        self.offset = PyroSample(
            prior=dist.Normal(0.0, 1.0).expand([1, 2])
        )
        self.base_beam = base_beam

    def forward(self):
        beam = self.base_beam()
        transformed_beam = beam.data + self.offset
        return Beam(
            transformed_beam, beam.p0c, beam.s, beam.mc2
        )


class BayesianReconstruction(PyroModule):
    def __init__(self):
        super().__init__()
        nn_transformer = NNTransform(2, 5, phase_space_dim=2, output_scale=10.0)
        nn_beam = InitialBeam(
            nn_transformer,
            torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)),
            1000,
            p0c=torch.tensor(63.0e6),
        )
        self.offset_beam = BayesianOffsetBeam(nn_beam)
        self.bins = torch.linspace(-5., 5., 100)
        self.bandwidth = 0.5 * (bins[1] - bins[0])

    def forward(self, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 0.1))
        d_size = 10 if y is None else y.shape[0]

        with pyro.plate("shots", d_size):
            beam_sample = offset_beam().data
            beam_img = histogram2d(
                beam_sample[:, 0],
                beam_sample[:, 1],
                self.bins,
                self.bins,
                self.bandwidth
            )

            with pyro.plate("x_bins", len(self.bins)):
                with pyro.plate("y_bins", len(self.bins)):
                    obs = pyro.sample("obs", dist.Normal(beam_img, sigma), obs=y)


offset_beam = BayesianOffsetBeam(nn_beam)

sample = offset_beam().data

# calculate the histogram
bins = torch.linspace(-5., 5., 100)
bandwidth = 0.5 * (bins[1] - bins[0])
img = histogram2d(sample[:, 0], sample[:, 1], bins, bins, bandwidth)

plt.imshow(img.detach().numpy())
plt.show()
