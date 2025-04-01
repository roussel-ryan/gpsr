from typing import Tuple, List

import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset
from cheetah.accelerator import Screen


class ObservableDataset(Dataset):
    """
    Define a PyTorch dataset to be passed to a TorchDataloader object for training.

    Parameters
    ----------
    parameters : Tensor
        Tensor of beamline parameters that correspond to data observations. Should
        have a shape of (B x N), B is a batch dimension and N is equal to the number of
        different parameters changed in the beamline. If parameters control which observation is being used,
        the shape should be (B x M x N) where M is the number of different observations. See Notes for an example.
    observations : Tuple[Tensor]
        Tuple of tensors contaning observed data. Tuple length should be M,
        or the number of different observations. Tensor shapes should be (B x D)
        where B is a batch dimension shape that corresponds to the `parameter`
        tensor, and D is an arbitrary shape corresponding to the dimensionality of
        the observable.

    Notes
    -----

    Example: 4D phase space reconstruction with a quadrupole scan on one screen
    using 5 quadrupole strengths. See below for the correct shapes of each input.

    >>> import torch
    >>> parameters = torch.rand((5, 1)) # B = (5), N = 1
    >>> observations = (
    >>>     torch.rand((5, 200, 200)),
    >>> )
    >>> dataset = ObservableDataset(parameters, observations)


    Example: 6D phase space reconstruction with 2 screens of various pixel sizes.
    Beamline parameters include the 2 dipole strengths (corresponding to each screen),
    2 transverse deflecting cavity strengths, and 5 quadrupole strenghs. See below for
    the correct shapes of each input. Note: that because the TCAV and quadrupole parameters
    do not change the measurement diagnostic, they are flattened into a single dimension.

    >>> import torch
    >>> parameters = torch.rand((10, 2, 3)) # B = 10, M = 2, N = 3
    >>> observations = (
    >>>     torch.rand((10, 200, 200)),
    >>>     torch.rand((10, 150, 150))
    >>> )
    >>> dataset = ObservableDataset(parameters, observations, n_observation_dims=1)

    """

    def __init__(
        self,
        parameters: Tensor,
        observations: Tuple[Tensor, ...],
    ):
        self.parameters = parameters
        self.observations = observations

        if not isinstance(observations, tuple):
            raise ValueError("observations must be passed as a tuple of tensors")

        # validate the input shapes
        if len(parameters.shape) == 2 or len(parameters.shape) == 3:
            # the leading dimension must match the number of observations

            if len(parameters.shape) == 3:
                # the leading dimension must match the number of observations
                if not parameters.shape[-2] == len(observations):
                    raise ValueError(
                        f"Number of observations {len(observations)} must match "
                        f"the [-2] dimension of parameters {parameters.shape[-2]}"
                    )
            for ele in self.observations:
                if not parameters.shape[0] == ele.shape[0]:
                    raise ValueError(
                        "Batch dimension of parameters must match batch dimension of observations"
                    )

        else:
            raise ValueError("parameters must be a 2D or 3D tensor")

    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, List[Tensor]]:
        """
        Get the parameters and observations for a given index.
        If the parameters are 3D we then batch along the
        """
        return (
            self.parameters[idx],
            [ele[idx] for ele in self.observations],
        )

    def plot_data(self):
        pass


DEFAULT_CONTOUR_LEVELS = [0.1, 0.5, 0.9]
DEFAULT_COLORMAP = "Greys"


class QuadScanDataset(ObservableDataset):
    def __init__(self, parameters: Tensor, observations: Tensor, screen: Screen):
        """
        Light wrapper dataset class for 4D phase space reconstructions with
        quadrupole. Checks for correct sizes of parameters and observations
        and provides a plotting utility.

        Parameters
        ----------
        parameters : Tensor
            Tensor of beamline parameters that correspond to data observations.
            Should have a shape of (K x 1) where K is the number of quadrupole strengths.
        observations : Tensor
            Tensor contaning observed images, where the tensor shape
            should be (K x bins x bins). First entry should be dipole off images.
        screen: Screen
            Cheetah screen object that corresponds to the observed images.

        """

        super().__init__(parameters, tuple([observations]))
        self.screen = screen

    def plot_data(
        self, overlay_data=None, overlay_kwargs: dict = None, filter_size: int = None
    ):
        # check overlay data size if specified
        if overlay_data is not None:
            assert isinstance(overlay_data, type(self))
            overlay_kwargs = {
                "levels": DEFAULT_CONTOUR_LEVELS,
                "cmap": DEFAULT_COLORMAP,
            } | (overlay_kwargs or {})

        parameters = self.parameters.flatten()
        n_k = len(parameters)
        fig, ax = plt.subplots(1, n_k, figsize=(n_k + 1, 1), sharex="all", sharey="all")

        xbins, ybins = self.screen.pixel_bin_centers
        xx = torch.meshgrid(xbins * 1e3, ybins * 1e3, indexing="ij")
        images = self.observations[0]

        for i in range(n_k):
            ax[i].pcolormesh(
                xx[0].numpy(),
                xx[1].numpy(),
                images[i] / images[i].max(),
                rasterized=True,
                vmax=1.0,
                vmin=0,
            )

            if overlay_data is not None:
                overlay_image = overlay_data.observations[0][i]
                if filter_size is not None:
                    overlay_image = gaussian_filter(overlay_image.numpy(), filter_size)

                ax[i].contour(
                    xx[0].numpy(),
                    xx[1].numpy(),
                    overlay_image / overlay_image.max(),
                    **overlay_kwargs,
                )

            ax[i].set_title(f"{parameters[i]:.1f}")
            ax[i].set_xlabel("x (mm)")

        ax[0].set_ylabel("y (mm)")
        ax[0].text(
            -0.1,
            1.1,
            "$k_1$ (1/m$^2$)",
            va="bottom",
            ha="right",
            transform=ax[0].transAxes,
        )

        return fig, ax


class SixDReconstructionDataset(ObservableDataset):
    def __init__(
        self,
        parameters: Tensor,
        observations: Tuple[Tensor, Tensor],
        screens: Tuple[Screen, Screen],
    ):
        """
        Light wrapper dataset class for 6D phase space reconstructions with quadrupole,
        dipole and TDC. Checks for correct sizes of parameters and observations.

        Parameters
        ----------
        parameters : Tensor
            Tensor of beamline parameters that correspond to data observations.
            Should elements along the last dimension should be ordered by (dipole
            strengths, TDC voltages, quadrupole focusing strengths) and should have a
            shape of (K x N x 2 x 3) where K is the number of quadrupole strengths.
        observations : Tuple[Tensor, Tensor]
            Tuple of tensors contaning observed images, where the tensor shapes
            should be (K x N x n_bins x n_bins). First entry should be dipole off
            images, second entry should be dipole on images.

        """

        # keep unflattened parameters and observations for visualization
        self.six_d_params = parameters
        self.six_d_observations = observations

        # flatten stuff here for the parent class
        parameters = self.six_d_params.clone().flatten(end_dim=-3)
        observations = tuple(
            [ele.clone().flatten(end_dim=-3) for ele in self.six_d_observations]
        )

        super().__init__(parameters, observations)
        self.screens = screens

    def plot_data(
        self,
        publication_size: bool = False,
        overlay_data=None,
        overlay_kwargs: dict = None,
        show_difference: bool = False,
        filter_size: int = 3,
    ):
        """
        Visualize dataset collected for 6-D phase space reconstructions

        """

        # check overlay data size if specified
        if overlay_data is not None:
            assert isinstance(overlay_data, type(self))
            okwargs = {
                "levels": DEFAULT_CONTOUR_LEVELS,
                "cmap": DEFAULT_COLORMAP,
            }
            okwargs.update(overlay_kwargs or {})

        n_k, n_v, n_g = self.six_d_params.shape[:-1]
        params = self.six_d_params
        images = self.six_d_observations

        # plot
        if publication_size:
            figsize = (7.5, (n_v + n_g) * 1.4)
            kwargs = {
                "top": 0.925,
                "bottom": 0.025,
                "right": 0.975,
                "hspace": 0.1,
                "wspace": 0.1,
            }
        else:
            figsize = ((n_k + 1) * 2, (n_v + n_g + 1) * 2)
            kwargs = {"right": 0.9}
        fig, ax = plt.subplots(
            n_v + n_g,
            n_k,
            figsize=figsize,
            gridspec_kw=kwargs,
            sharex="all",
            sharey="all",
        )

        # ax[0, 0].set_axis_off()
        ax[0, 0].text(
            -0.1,
            1.1,
            "$k_1$ (1/m$^2$)",
            va="bottom",
            ha="right",
            transform=ax[0, 0].transAxes,
        )
        for i in range(n_k):
            # ax[0, i].set_axis_off()
            ax[0, i].text(
                0.5,
                1.1,
                f"{params[i, 0, 0, 0]:.1f}",
                va="bottom",
                ha="center",
                transform=ax[0, i].transAxes,
            )
            for j in range(n_g):
                for k in range(n_v):
                    px_bin_centers = self.screens[j].pixel_bin_centers
                    px_bin_centers = px_bin_centers[0] * 1e3, px_bin_centers[1] * 1e3
                    row_number = 2 * j + k

                    if show_difference and overlay_data is not None:
                        # if flags are specified plot the difference
                        diff = torch.abs(
                            images[j][i, k] - overlay_data.six_d_observations[j][i, k]
                        )
                        ax[row_number, i].pcolor(
                            *px_bin_centers,
                            diff,
                            rasterized=True,
                        )
                        ax[row_number, i].text(
                            0.01,
                            0.99,
                            f"{torch.sum(diff):.2}",
                            color="white",
                            transform=ax[row_number, i].transAxes,
                            ha="left",
                            va="top",
                            fontsize=8,
                        )

                    else:
                        ax[row_number, i].pcolor(
                            *px_bin_centers,
                            images[j][i, k] / images[j][i, k].max(),
                            rasterized=True,
                            vmax=1.0,
                            vmin=0,
                        )

                        if overlay_data is not None:
                            overlay_image = overlay_data.six_d_observations[j][
                                i, k
                            ].numpy()
                            if filter_size is not None:
                                overlay_image = gaussian_filter(
                                    overlay_image, filter_size
                                )

                            ax[row_number, i].contour(
                                *px_bin_centers,
                                overlay_image / overlay_image.max(),
                                **okwargs,
                            )

                    if k == 0:
                        v_lbl = "off"
                    else:
                        v_lbl = "on"
                    if j == 0:
                        g_lbl = "off"
                    else:
                        g_lbl = "on"

                    if i == 0:
                        ax[row_number, 0].text(
                            -0.6,
                            0.5,
                            f"T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}",
                            va="center",
                            ha="right",
                            transform=ax[row_number, 0].transAxes,
                        )
        # fig.tight_layout()
        for ele in ax[-1]:
            ele.set_xlabel("x (mm)")
        for ele in ax[:, 0]:
            ele.set_ylabel("y (mm)")
        return fig, ax
