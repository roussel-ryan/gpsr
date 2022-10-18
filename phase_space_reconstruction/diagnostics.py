import torch
from torch.nn import Module

from phase_space_reconstruction.histogram import histogram2d


class ImageDiagnostic(Module):
    def __init__(self, bins, bandwidth=None):
        """

        :param bins: 1D tensor of bin edges for image diagnostic
        """

        super(ImageDiagnostic, self).__init__()

        self.register_buffer("bins", bins)
        self.register_buffer("resolution", bins[1] - bins[0])
        self.register_buffer(
            "bandwidth", self.resolution if bandwidth is None else bandwidth
        )
        self.register_buffer(
            "mesh",
            torch.cat(
                [ele.unsqueeze(0) for ele in torch.meshgrid(self.bins, self.bins)],
                dim=0,
            ),
        )

    def calculate_images(self, x_coords, y_coords):
        """
        :param x_coords: (`batch_shape` x N) tensor of x coordinates of N particles
        :param y_coords: (`batch_shape` x N) tensor of x coordinates of N particles
        :return: ('batch_shape' x M x M) tensor with pixel intensities for M x M images
        """
        if not x_coords.shape == y_coords.shape:
            raise ValueError("x,y coords must be the same shape")

        if len(x_coords.shape) == 1:
            raise ValueError("coords must be at least 2D")
        return histogram2d(x_coords, y_coords, self.bins, self.bandwidth)
