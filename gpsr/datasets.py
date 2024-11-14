from typing import Tuple, List

import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


class ObservableDataset(Dataset):
    """
    Define a PyTorch dataset to be passed to a TorchDataloader object for training.

    Parameters
    ----------
    parameters : Tensor
        Tensor of beamline parameters that correspond to data observations. Should
        have a shape of (M x B x N) where M is the number of different observations,
        B is a batch dimension of arbitrary shape, and N is equal to the number of
        different parameters changed in the beamline. See Notes for an example.
    observations : Tuple[Tensor]
        Tuple of tensors contaning observed data. Tuple length should be M,
        or the number of different observations. Tensor shapes should be (B x D)
        where B is a batch dimension shape that corresponds to the `parameter`
        tensor, and D is an arbitrary shape corresponding to the dimensionality of
        the observable.

    Notes
    -----
    Example: 6D phase space reconstruction with 2 screens of various pixel sizes.
    Beamline parameters include the 2 dipole strengths (corresponding to each screen),
    2 transverse deflecting cavity strengths, and 5 quadrupole strenghs. See below for
    the correct shapes of each input.

    >>> import torch
    >>> parameters = torch.rand((2, 2, 5, 3)) # M = 2, B = (2, 5), N = 3
    >>> observations = (
    >>>     torch.rand((2, 5, 200, 200)),
    >>>     torch.rand((2, 5, 150, 150))
    >>> )
    >>> dataset = ObservableDataset(parameters, observations)

    """

    def __init__(self, parameters: Tensor, observations: Tuple[Tensor, ...]):
        self.parameters = parameters
        self.observations = observations

        if not isinstance(observations, tuple):
            raise ValueError("observations must be passed as a tuple of tensors")

        if len(self.observations) > 1:
            assert len(self.observations) == self.parameters.shape[0]

            # we have to flatten any batch dimensions B for batching purposes
            batch_shape = self.parameters.shape[1:-1]
            self._flattened_parameters = torch.flatten(
                parameters, start_dim=1, end_dim=-2
            )
            self._flattened_observations = tuple(
                [
                    torch.flatten(ele, end_dim=len(batch_shape) - 1)
                    for ele in self.observations
                ]
            )
        else:
            self._flattened_parameters = parameters
            self._flattened_observations = observations

    def __len__(self):
        return self._flattened_parameters.shape[1]

    def __getitem__(self, idx) -> (Tensor, List[Tensor]):
        return (
            self._flattened_parameters[:, idx],
            [ele[idx] for ele in self._flattened_observations],
        )

    def plot_data(self):
        pass


class FourDReconstructionDataset(ObservableDataset):
    def __init__(self, parameters: Tensor, observations: Tensor, bins: Tensor):
        """
        Light wrapper dataset class for 4D phase space reconstructions with
        quadrupole. Checks for correct sizes of parameters and observations
        and provides a plotting utility.

        Parameters
        ----------
        parameters : Tensor
            Tensor of beamline parameters that correspond to data observations.
            Should elements along the last dimension should be ordered by (dipole
            strengths, TDC voltages, quadrupole focusing strengths) and should have a
            shape of (K x N) where K is the number of quadrupole strengths.
        observations : Tensor
            Tensor contaning observed images, where the tensor shapes
            should be (K x bins x bins). First entry should be dipole off images.
        bins: Tensor
            Tensor of 1-D bin locations for each image set. Assumes square images with
            the same bin locations.

        """

        super().__init__(parameters.unsqueeze(0), tuple(observations.unsqueeze(0)))
        self.bins = bins

    def plot_data(self, overlay_data=None, overlay_kwargs: dict = None):
        # check overlay data size if specified
        if overlay_data is not None:
            assert isinstance(overlay_data, type(self))
            overlay_kwargs = overlay_kwargs or {
                "levels": [0.1, 0.25, 0.5, 0.75, 0.9],
                "cmap": "Greys",
            }

        parameters = self.parameters[0]
        n_k = len(parameters)
        fig, ax = plt.subplots(1, n_k, figsize=(n_k + 1, 1), sharex="all", sharey="all")

        xx = torch.meshgrid(self.bins * 1e3, self.bins * 1e3, indexing="ij")
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
                img = gaussian_filter(images[i].numpy(), 3)

                ax[i].contour(
                    xx[0].numpy(), xx[1].numpy(), img / img.max(), **overlay_kwargs
                )

            ax[i].set_title(f"{parameters[i]:.1f}")
            ax[i].set_xlabel("x (mm)")
            # ax[0].text(0.5, 0.5, f"img 1", va="center", ha="center")

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
        bins: Tuple[Tensor, Tensor],
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
            shape of (2 x 2 x K x N) where K is the number of quadrupole strengths.
        observations : Tuple[Tensor, Tensor]
            Tuple of tensors contaning observed images, where the tensor shapes
            should be (2 x K x n_bins x n_bins). First entry should be dipole off
            images.
        bins: Tuple[Tensor, Tensor]
            Tuple of 1-D bin locations for each image set. Assumes square images with
            the same bin locations.

        """

        assert parameters.shape[:2] == torch.Size([2, 2])
        assert len(observations) == 2

        super().__init__(parameters, observations)
        self.bins = bins

    def plot_data(
        self,
        publication_size: bool = False,
        overlay_data=None,
        overlay_kwargs: dict = None,
        show_difference: bool = False
    ):
        """
        Visualize dataset collected for 6-D phase space reconstructions

        """

        # check overlay data size if specified
        if overlay_data is not None:
            assert isinstance(overlay_data, type(self))
            overlay_kwargs = overlay_kwargs or {
                "levels": [0.1, 0.25, 0.5, 0.75, 0.9],
                "cmap": "Greys",
            }

        n_g, n_v, n_k = self.parameters.shape[:-1]
        params = self.parameters
        images = self.observations
        bins = self.bins

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
                f"{params[0, 0, i, -1]:.1f}",
                va="bottom",
                ha="center",
                transform=ax[0, i].transAxes,
            )
            for k in range(n_v):
                for j in range(n_g):
                    xx = torch.meshgrid(bins[j] * 1e3, bins[j] * 1e3)
                    row_number = 2 * j + k

                    if show_difference and overlay_data is not None:
                        # if flags are specified plot the difference
                        diff = torch.abs(
                            images[j][k, i] - overlay_data.observations[j][k, i]
                        )
                        ax[row_number, i].pcolormesh(
                            xx[0].numpy(),
                            xx[1].numpy(),
                            diff,
                            rasterized=True,
                        )
                        ax[row_number, i].text(
                            0.01,
                            0.99, f'{torch.sum(diff):.2}',
                            color='white',
                            transform=ax[row_number, i].transAxes,
                            ha='left', va='top', fontsize=8
                        )

                    else:
                        ax[row_number, i].pcolormesh(
                            xx[0].numpy(),
                            xx[1].numpy(),
                            images[j][k, i] / images[j][k, i].max(),
                            rasterized=True,
                            vmax=1.0,
                            vmin=0,
                        )

                        if overlay_data is not None:
                            img = gaussian_filter(
                                overlay_data.observations[j][k, i].numpy(), 3
                            )

                            ax[row_number, i].contour(
                                xx[0].numpy(),
                                xx[1].numpy(),
                                img / img.max(),
                                **overlay_kwargs,
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
