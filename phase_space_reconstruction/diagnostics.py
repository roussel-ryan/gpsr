import torch
from bmadx.bmad_torch.track_torch import Beam
from torch.nn import Module

from phase_space_reconstruction.histogram import histogram2d


class ImageDiagnostic(Module):
    def __init__(
        self,
        bins_x: torch.Tensor,
        bins_y: torch.Tensor,
        bandwidth: torch.Tensor,
        x="x",
        y="y",
    ):
        """
        Parameters
        ----------
        bins_x : Tensor
            A 'n' mesh of pixel centers that correspond to the physical diagnostic.

        bins_y : Tensor
            A 'm' mesh of pixel centers that correspond to the physical diagnostic.

        bandwidth : Tensor
            Bandwidth uses for kernel density estimation

        x : str, optional
            Beam attribute coorsponding to the horizontal image axis. Default: `x`

        y : str, optional
            Beam attribute coorsponding to the vertical image axis. Default: `y`
        """

        super(ImageDiagnostic, self).__init__()
        self.x = x
        self.y = y

        self.register_buffer("bins_x", bins_x)
        self.register_buffer("bins_y", bins_y)
        self.register_buffer("bandwidth", bandwidth)

    def forward(self, beam: Beam):
        x_vals = getattr(beam, self.x)
        y_vals = getattr(beam, self.y)
        if not x_vals.shape == y_vals.shape:
            raise ValueError("x,y coords must be the same shape")

        if len(x_vals.shape) == 1:
            raise ValueError("coords must be at least 2D")

        return histogram2d(x_vals, y_vals, self.bins_x, self.bins_y, self.bandwidth)
