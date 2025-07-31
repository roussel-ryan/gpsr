from typing import Tuple, List, Union

import numpy as np
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
        the observable. The images must follow the matrix convention,
        where axis -2 is Y and axis -1 is X.

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
            The images must follow the matrix convention, where axis -2 is Y and
            axis -1 is X.
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

        px_bin_centers = self.screen.pixel_bin_centers
        px_bin_centers = px_bin_centers[0] * 1e3, px_bin_centers[1] * 1e3
        images = self.observations[0]

        for i in range(n_k):
            ax[i].pcolormesh(
                *px_bin_centers,
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
                    *px_bin_centers,
                    overlay_image / overlay_image.max(),
                    **overlay_kwargs,
                )

            ax[i].set_title(f"{parameters[i]:.1f}")
            ax[i].set_xlabel("x (mm)")
            ax[i].set_aspect("equal")

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
        six_d_parameters: Tensor,
        six_d_observations: Tuple[Tensor, Tensor],
        screens: Tuple[Screen, Screen],
    ):
        """
        Light wrapper dataset class for 6D phase space reconstructions with
        quadrupole, dipole, and TDC. Checks for correct sizes of parameters
        and observations.

        Parameters
        ----------
        six_d_parameters : Tensor
            Tensor of beamline parameters that correspond to data observations.
            Shape should be (K x N x 2 x 3) where K is the number of quadrupole
            strengths, N the number of TDC voltages, and 2 is the number of
            dipole strengths. The last dimension should be ordered as
            (quadrupole focusing strengths, TDC voltages, dipole strengths).

        six_d_observations : Tuple[Tensor, Tensor]
            Tuple of tensors containing observed images, where the tensor shapes
            should be (K x N x n_bins x n_bins). Here, K is the number of
            quadrupole strengths, and N is the number of TDC voltages. The first
            entry should be dipole-off images, and the second entry should be
            dipole-on images. The images must follow the matrix convention,
            where axis -2 is Y and axis -1 is X.

        screens: Tuple[Screen, Screen]
            Tuple of cheetah screen objects that corresponds to the observed images.

        Notes
        -----

        Only squared images are supported for now.
        """

        # keep unflattened parameters and observations for visualization
        self.six_d_parameters = six_d_parameters
        self.six_d_observations = six_d_observations

        # flatten stuff here for the parent class
        parameters = self.six_d_parameters.clone().flatten(end_dim=-3)
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

        n_k, n_v, n_g = self.six_d_parameters.shape[:-1]
        params = self.six_d_parameters
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
                        ax[row_number, i].pcolormesh(
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
                        ax[row_number, i].pcolormesh(
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

                    ax[row_number, i].set_aspect("equal")
        # fig.tight_layout()
        for ele in ax[-1]:
            ele.set_xlabel("x (mm)")
        for ele in ax[:, 0]:
            ele.set_ylabel("y (mm)")
        return fig, ax


class FiveDReconstructionDataset(ObservableDataset):
    def __init__(
        self,
        parameters: Tensor,
        observations: Tuple[Tensor, Tensor],
        screens: Tuple[Screen, Screen],
    ):
        """
        parameters:
        """

        super().__init__(parameters, tuple(observations))
        self.screens = screens

    def plot_data(self):
        fig, ax = plt.subplots(
            2,
            self.observations[0].shape[0],
            figsize=(2 * self.observations[0].shape[0], 2 * 2),
            sharex="all",
            sharey="all",
        )
        for screen in range(2):
            for k1 in range(self.observations[0].shape[0]):
                px_bin_centers = self.screens[screen].pixel_bin_centers
                px_bin_centers = px_bin_centers[0] * 1e3, px_bin_centers[1] * 1e3
                ax[screen, k1].pcolormesh(
                    *px_bin_centers,
                    self.observations[screen][k1] / self.observations[screen][k1].max(),
                    rasterized=True,
                    vmax=1.0,
                    vmin=0,
                )
                ax[screen, k1].set_aspect("equal")
                ax[screen, k1].set_title(f"{self.parameters[k1][screen][-1]:.2f}")
        for ele in ax[-1]:
            ele.set_xlabel("x (mm)")
        for ele in ax[:, 0]:
            ele.set_ylabel("y (mm)")
        return fig, ax


def split_dataset(
    dataset: Union[SixDReconstructionDataset, QuadScanDataset],
    train_k_ids: np.ndarray,
    test_k_ids: np.ndarray = None,
) -> Tuple[
    Union[SixDReconstructionDataset, QuadScanDataset],
    Union[SixDReconstructionDataset, QuadScanDataset],
]:
    """
    Splits a dataset into training and testing subsets based on provided indices.

    Args:
        dataset (Union[SixDReconstructionDataset, QuadScanDataset]):
            The dataset to be split. It can be either a SixDReconstructionDataset
            or a QuadScanDataset.
        train_k_ids (np.ndarray):
            An array of indices specifying the samples to include in the training subset.
        test_k_ids (np.ndarray, optional):
            An array of indices specifying the samples to include in the testing subset.
            If not provided, the testing subset will include all indices not in `train_k_ids`.

    Returns:
        Tuple[Union[SixDReconstructionDataset, QuadScanDataset],
              Union[SixDReconstructionDataset, QuadScanDataset]]:
            A tuple containing the training dataset and the testing dataset, both of the
            same type as the input dataset.

    Raises:
        ValueError: If the input dataset is not of type SixDReconstructionDataset or
                    QuadScanDataset.
    """

    def generate_test_indices(all_k_ids, train_k_ids, test_k_ids=None):
        """
        Generate test indices based on the provided train indices and optional test indices.
        If test_k_ids is None, compute the difference between all_k_ids and train_k_ids.
        """
        if test_k_ids is None:
            return np.setdiff1d(all_k_ids, train_k_ids)
        return test_k_ids

    if isinstance(dataset, SixDReconstructionDataset):
        all_k_ids = np.arange(dataset.six_d_parameters.shape[0])
        test_k_ids_copy = generate_test_indices(all_k_ids, train_k_ids, test_k_ids)

        train_dataset = SixDReconstructionDataset(
            six_d_parameters=dataset.six_d_parameters[train_k_ids],
            six_d_observations=tuple(
                observation[train_k_ids] for observation in dataset.six_d_observations
            ),
            screens=dataset.screens,
        )

        test_dataset = SixDReconstructionDataset(
            six_d_parameters=dataset.six_d_parameters[test_k_ids_copy],
            six_d_observations=tuple(
                observation[test_k_ids_copy]
                for observation in dataset.six_d_observations
            ),
            screens=dataset.screens,
        )

    elif isinstance(dataset, QuadScanDataset):
        all_k_ids = np.arange(dataset.parameters.shape[0])
        test_k_ids_copy = generate_test_indices(all_k_ids, train_k_ids, test_k_ids)

        train_dataset = QuadScanDataset(
            parameters=dataset.parameters[train_k_ids],
            observations=dataset.observations[0][train_k_ids],
            screen=dataset.screen,
        )

        test_dataset = QuadScanDataset(
            parameters=dataset.parameters[test_k_ids_copy],
            observations=dataset.observations[0][test_k_ids_copy],
            screen=dataset.screen,
        )

    else:
        raise ValueError("Unknown dataset type")

    return train_dataset, test_dataset
