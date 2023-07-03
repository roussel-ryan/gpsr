import torch
from torch.nn import Module

from phase_space_reconstruction.histogram import histogram2d


class ImageDiagnostic(Module):
    def __init__(self, bins, bandwidth=None, x="x", y="y"):
        """

        :param bins: 1D tensor of bin edges for image diagnostic
        """

        super(ImageDiagnostic, self).__init__()
        self.x = x
        self.y = y

        self.register_buffer("bins", bins)
        self.register_buffer("resolution", bins[1] - bins[0])
        self.register_buffer(
            "bandwidth", self.resolution if bandwidth is None else bandwidth
        )
        self.register_buffer(
            "mesh",
            torch.cat(
                [ele.unsqueeze(0) for ele in torch.meshgrid(self.bins, self.bins, indexing='ij')],
                dim=0,
            ),
        )


    def forward(self, beam):
        """
        :param beam:
             :return: ('batch_shape' x M x M) tensor with pixel intensities for M x M images
        """
        x_vals = getattr(beam, self.x)
        y_vals = getattr(beam, self.y)
        if not x_vals.shape == y_vals.shape:
            raise ValueError("x,y coords must be the same shape")

        if len(x_vals.shape) == 1:
            raise ValueError("coords must be at least 2D")
        
        return histogram2d(x_vals, y_vals, self.bins, self.bandwidth)
