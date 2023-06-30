import torch
from torch.nn import Module
from torch import Tensor

from phase_space_reconstruction.histogram import KDEGaussian
from bmadx.bmad_torch.track_torch import Beam


class ImageDiagnostic(Module):
    def __init__(self, mesh: Tensor, bandwidth: float, x="x", y="y"):
        """

        Parameters
        ----------
        mesh : Tensor
            A 'n x m' mesh of pixel centers that correspond to the physical diagnostic.

        bandwidth : float
            Bandwidth uses for kernel density estimation

        x : str, optional
            Beam attribute coorsponding to the horizontal image axis. Default: `x`

        y : str, optional
            Beam attribute coorsponding to the vertical image axis. Default: `y`
        """
        super(ImageDiagnostic, self).__init__()
        self.x = x
        self.y = y

        self.register_buffer("mesh", mesh)
        self.kde_calculator = KDEGaussian(bandwidth, locations=mesh)

    def forward(self, beam: Beam):
        x_vals = getattr(beam, self.x)
        y_vals = getattr(beam, self.y)
        if not x_vals.shape == y_vals.shape:
            raise ValueError("x,y coords must be the same shape")

        if len(x_vals.shape) == 1:
            raise ValueError("coords must be at least 2D")

        beam_vals = torch.stack((x_vals, y_vals))
        
        return self.kde_calculator(beam_vals)
