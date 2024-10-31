from typing import Tuple, List

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
        self.observations = observations
        self.parameters = parameters

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx) -> (Tensor, List[Tensor]):
        return self.parameters[idx], [ele[idx] for ele in self.observations]
