import torch
from torch import nn, Tensor


class KDEGaussian(nn.Module):
    def __init__(self, bandwidth: float, locations: Tensor = None):
        """
        Object to calculate a differentiable Kernel Desity Estimation at an arbitrary
        set of locations in space.

        Parameters
        ----------
        bandwidth: float
            Bandwidth of the kernel density estimator.

        locations: Tensor
            A `m x n` tensor where `m` is the number of KDE evaluation points at
            points in `n`-dim space.
        """

        super(KDEGaussian, self).__init__()
        self.bandwidth = bandwidth
        self.locations = locations

    def forward(self, samples: Tensor, locations=None) -> Tensor:
        """
            Object to calculate a differentiable Kernel Desity Estimation at an arbitrary
            set of locations in space.

            Parameters
            ----------
            samples: Tensor
                A `batch_size x n` tensor of sample points to calculate the KDE on.
            locations: Tensor
                A `m x n` tensor where `m` is the number of KDE evaluation points at
                points in `n`-dim space.


            Returns
            -------
            result: Tensor
                A `batch_size x m' tensor of KDE values calculated at `m` points for
                a given number of batches.

        """

        assert samples.shape[-1] == locations.shape[-1]

        # make copies of all samples for each location
        all_samples = samples.reshape(samples.shape + (1,) * len(locations.shape[:-1]))
        diff = torch.norm(
            all_samples - torch.movedim(locations, -1, 0),
            dim=-len(locations.shape[:-1]) - 1,
        )
        out = (-diff ** 2 / (2.0 * self.bandwidth ** 2)).exp().sum(dim=len(
            samples.shape)-2)
        norm = out.flatten(start_dim=len(locations.shape)-2).sum(dim=-1)
        return out / norm.reshape(-1, *(1,)*(len(locations.shape)-1))
