from typing import Literal
import multiprocessing as mp
import threading
import os
import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from gpsr.modeling import GPSR
from gpsr.train import train_gpsr
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from lightning.pytorch.loggers import CSVLogger
from scipy.ndimage import gaussian_filter

from cheetah.particles import ParticleBeam


def reinitialize_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
    elif hasattr(m, "weight") and hasattr(m.weight, "data"):
        if len(m.weight.shape) >= 2:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.uniform_(m.weight, -0.1, 0.1)
    if hasattr(m, "bias") and m.bias is not None:
        torch.nn.init.zeros_(m.bias)


def _train_single_model_thread(
    model_idx,
    gpsr_model,
    train_dataloader,
    n_epochs,
    lr,
    log_name,
    checkpoint_period_epochs,
    results_dict,
    gpu_id=None,
    **kwargs,
):
    """Helper function to train a single model on a specific GPU using threading."""
    try:
        # Set unique random seed for this model to ensure different initialization
        import random

        unique_seed = hash(
            (model_idx, os.getpid(), threading.current_thread().ident)
        ) % (2**32)
        torch.manual_seed(unique_seed)
        np.random.seed(unique_seed % (2**31))
        random.seed(unique_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(unique_seed)
            torch.cuda.manual_seed_all(unique_seed)

        # Set GPU device if specified
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            # Add accelerator and devices to kwargs for Lightning
            kwargs.update({"accelerator": "gpu", "devices": [gpu_id]})

        print(
            f"Training model {model_idx + 1} on GPU {gpu_id if gpu_id is not None else 'default'} with seed {unique_seed}..."
        )

        # Define logger
        logger = CSVLogger("logs", name=log_name + f"/model_{model_idx}")

        # Create a deep copy of the model to avoid sharing state
        model_copy = copy.deepcopy(gpsr_model)

        # Reinitialize the model parameters to ensure uniqueness
        model_copy.apply(reinitialize_weights)

        model = train_gpsr(
            model_copy,
            train_dataloader,
            n_epochs=n_epochs,
            lr=lr,
            logger=logger,
            checkpoint_period_epochs=checkpoint_period_epochs,
            **kwargs,
        )

        results_dict[model_idx] = model
        print(f"Model {model_idx + 1} training completed successfully")

    except Exception as e:
        print(f"Model {model_idx + 1} training failed with exception: {e}")
        results_dict[model_idx] = e


def train_ensemble(
    gpsr_model: GPSR,
    train_dataloader: torch.utils.data.DataLoader,
    n_models: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    log_name: str = "gpsr_ensemble",
    checkpoint_period_epochs: int = 100,
    parallel_training: Literal["auto", True, False] = "auto",
    max_workers: int = None,
    gpu_ids: list[int] = None,
    **kwargs,
):
    """
    Train an ensemble of GPSR models with automatic fallback to sequential training.

    Parameters
    ----------
    gpsr_model :
        The GPSR model configuration.
    train_dataloader :
        DataLoader for training data.
    n_models : int, optional
        Number of models in the ensemble. Default is 5.
    n_epochs : int, optional
        Number of training epochs per model. Default is 100.
    lr : float, optional
        Learning rate. Default is 1e-3.
    log_name : str, optional
        Base name for logging. Default is "gpsr_ensemble".
    checkpoint_period_epochs : int, optional
        Period for saving checkpoints. Default is 100.
    parallel_training : str or bool, optional
        Training mode. Options:
        - "auto" (default): Automatically choose parallel if multiple GPUs available, else sequential
        - True: Force parallel training (will raise error if insufficient GPUs)
        - False: Force sequential training
    max_workers : int, optional
        Maximum number of parallel threads. If None, uses min(n_models, available_gpus).
    gpu_ids : list[int], optional
        List of GPU IDs to use. If None, automatically detects available GPUs.
    **kwargs :
        Additional arguments passed to train_gpsr.

    Returns
    -------
    list
        List of trained models.
    """

    def _train_sequential():
        """Sequential training implementation."""
        models = []
        print("Training models sequentially...")

        # check gpsr_model type
        if not isinstance(gpsr_model, GPSR):
            raise ValueError("gpsr_model must be an instance of GPSR")

        for i in range(n_models):
            # Set unique random seed for this model to ensure different initialization
            import random

            unique_seed = hash((i, os.getpid(), "sequential")) % (2**32)
            torch.manual_seed(unique_seed)
            np.random.seed(unique_seed % (2**31))
            random.seed(unique_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(unique_seed)
                torch.cuda.manual_seed_all(unique_seed)

            print(f"Training model {i + 1}/{n_models} with seed {unique_seed}...")

            # define logger
            logger = CSVLogger("logs", name=log_name + f"/model_{i}")

            # Create a deep copy and reinitialize
            model_copy = copy.deepcopy(gpsr_model)

            # Reinitialize the model parameters to ensure uniqueness
            model_copy.apply(reinitialize_weights)

            model = train_gpsr(
                model_copy,
                train_dataloader,
                n_epochs=n_epochs,
                lr=lr,
                logger=logger,
                checkpoint_period_epochs=checkpoint_period_epochs,
                **kwargs,
            )
            models.append(model)

        return models

    def _train_parallel():
        """Parallel training implementation."""
        # Use threading instead of multiprocessing to avoid tensor serialization issues
        results_dict = {}
        threads = []

        # Create semaphore to limit concurrent training
        semaphore = threading.Semaphore(max_workers)

        def thread_wrapper(model_idx, gpu_id, model):
            with semaphore:
                _train_single_model_thread(
                    model_idx,
                    model,
                    train_dataloader,
                    n_epochs,
                    lr,
                    log_name,
                    checkpoint_period_epochs,
                    results_dict,
                    gpu_id,
                    **kwargs,
                )

        print(
            f"Training {n_models} models in parallel using {max_workers} threads on GPUs {effective_gpu_ids[:max_workers]}"
        )

        # Start all training threads
        for i in range(n_models):
            gpu_id = effective_gpu_ids[i % len(effective_gpu_ids)]
            thread = threading.Thread(
                target=thread_wrapper, args=(i, gpu_id, gpsr_model)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results and check for errors
        models = [None] * n_models
        for i in range(n_models):
            if i in results_dict:
                if isinstance(results_dict[i], Exception):
                    raise results_dict[i]
                models[i] = results_dict[i]
            else:
                raise RuntimeError(
                    f"Model {i + 1} training failed - no result returned"
                )

        return models

    # Determine training mode and GPU availability
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    # Determine effective GPU IDs
    if gpu_ids is None:
        effective_gpu_ids = list(range(num_gpus)) if cuda_available else []
    else:
        effective_gpu_ids = gpu_ids

    # Determine if parallel training is feasible and desired
    can_parallel = cuda_available and num_gpus >= 2 and len(effective_gpu_ids) >= 2

    if parallel_training == "auto":
        use_parallel = can_parallel
        if not can_parallel and cuda_available:
            print(
                f"Auto-mode: Only {num_gpus} GPU(s) available. Falling back to sequential training."
            )
        elif not cuda_available:
            print("Auto-mode: CUDA not available. Using sequential training.")
    elif parallel_training is True:
        if not can_parallel:
            if not cuda_available:
                raise RuntimeError(
                    "CUDA is not available. Parallel training requires multiple GPUs."
                )
            else:
                raise RuntimeError(
                    f"Only {num_gpus} GPU(s) available. Parallel training requires at least 2 GPUs."
                )
        use_parallel = True
    else:  # parallel_training is False
        use_parallel = False

    # Set max_workers if not provided
    if use_parallel and max_workers is None:
        max_workers = min(n_models, len(effective_gpu_ids))
    elif use_parallel:
        max_workers = min(max_workers, len(effective_gpu_ids))

    # Execute training
    if use_parallel:
        return _train_parallel()
    else:
        return _train_sequential()


def train_ensemble_distributed(
    gpsr_lattice,
    train_dataloader,
    n_models=5,
    n_epochs=100,
    lr=1e-3,
    log_name="gpsr_ensemble",
    checkpoint_period_epochs=100,
    strategy="ddp",
    **kwargs,
):
    """
    Alternative approach: Train models using PyTorch Lightning's distributed training.
    This trains each model using all available GPUs with data parallelism.

    Parameters
    ----------
    strategy : str, optional
        PyTorch Lightning strategy for distributed training. Options include:
        - "ddp" : Distributed Data Parallel
        - "ddp_spawn" : DDP with spawn
        - "fsdp" : Fully Sharded Data Parallel
        Default is "ddp".
    """
    models = []

    # Set up distributed training configuration
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available for distributed training.")

    # Add distributed training parameters to kwargs
    distributed_kwargs = kwargs.copy()
    distributed_kwargs.update(
        {
            "strategy": strategy,
            "accelerator": "gpu",
            "devices": num_gpus if num_gpus > 1 else 1,
        }
    )

    for i in range(n_models):
        print(
            f"Training model {i + 1}/{n_models} with distributed training on {num_gpus} GPUs..."
        )

        # define logger
        logger = CSVLogger("logs", name=log_name + f"/model_{i}")

        p0c = 43.36e6  # reference momentum in eV/c

        model = train_gpsr(
            GPSR(
                NNParticleBeamGenerator(
                    10000, p0c, transformer=NNTransform(2, 20, output_scale=1e-2)
                ),
                gpsr_lattice,
            ),
            train_dataloader,
            n_epochs=n_epochs,
            lr=lr,
            logger=logger,
            checkpoint_period_epochs=checkpoint_period_epochs,
            **distributed_kwargs,
        )
        models.append(model)

    return models


def compute_mean_and_confidence_interval(
    histograms: np.ndarray,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of histograms, compute the mean and confidence interval over the first axis.

    Parameters
    ----------
    histograms: np.ndarray
        A 2D array of histograms, where each row is a histogram.
    lower_quantile: float, optional
        The lower percentile for the confidence interval. Default is 0.05.
    upper_quantile: float, optional
        The upper percentile for the confidence interval. Default is 0.95.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the mean histogram and the normalized confidence width.

    """
    mean_histogram = torch.mean(histograms, axis=0)
    lower_bound = torch.quantile(histograms, lower_quantile, axis=0)
    upper_bound = torch.quantile(histograms, upper_quantile, axis=0)
    normalized_confidence_width = mean_histogram / (upper_bound - lower_bound)
    normalized_confidence_width[normalized_confidence_width == torch.nan] = 0.0

    return mean_histogram, normalized_confidence_width


def compute_distribution_statistics(
    beams: list[ParticleBeam],
    x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    bins: int = 100,
    bin_ranges: tuple[tuple[float]] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    smoothing_factor: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a list of beams, compute the mean and confidence interval for the specified dimensions.

    Parameters
    ----------
    beams: list[ParticleBeam]
        List of ParticleBeam objects.
    x_dimension: str
        The x dimension to compute the mean and confidence interval for.
        Options are "x", "px", "y", "py", "tau", "p".
    y_dimension: str
        The y dimension to compute the mean and confidence interval for.
        Options are "x", "px", "y", "py", "tau", "p".
    bins: int, optional
        Number of bins to use for the histogram. Default is 100.
    bin_ranges: tuple[tuple[float]], optional
        The ranges for the x and y dimensions. Default is None, which means the ranges will be computed from the data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the x and y centers of the histogram, the mean, and the confidence interval.

    """
    # get maximum and minimum values for the x and y dimensions if not provided
    if bin_ranges is None:
        x_min = min([getattr(beam, x_dimension).min().detach() for beam in beams])
        x_max = max([getattr(beam, x_dimension).max().detach() for beam in beams])
        y_min = min([getattr(beam, y_dimension).min().detach() for beam in beams])
        y_max = max([getattr(beam, y_dimension).max().detach() for beam in beams])
        bin_ranges = ((x_min, x_max), (y_min, y_max))

    histograms = []

    for beam in beams:
        # get the histogram for each beam
        histogram, x_edges, y_edges = np.histogram2d(
            getattr(beam, x_dimension).cpu().detach().numpy(),
            getattr(beam, y_dimension).cpu().detach().numpy(),
            bins=bins,
            range=bin_ranges,
        )
        if smoothing_factor is not None:
            histogram = gaussian_filter(histogram, sigma=smoothing_factor)

        histograms.append(histogram)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    mean_histogram, normalized_confidence_width = compute_mean_and_confidence_interval(
        torch.tensor(histograms),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    return (
        x_centers,
        y_centers,
        mean_histogram,
        normalized_confidence_width,
    )


def plot_2d_distribution(
    beams: list[ParticleBeam],
    x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    bins: int = 100,
    bin_ranges: tuple[tuple[float]] | None = None,
    ci_kws: dict | None = None,
    density_kws: dict | None = None,
    ax: plt.Axes | None = None,
    smoothing_factor: float = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    x_centers, y_centers, mean_histogram, normalized_confidence_width = (
        compute_distribution_statistics(
            beams,
            x_dimension,
            y_dimension,
            bins=bins,
            bin_ranges=bin_ranges,
            smoothing_factor=smoothing_factor,
        )
    )

    c = ax[0].pcolormesh(
        x_centers,
        y_centers,
        mean_histogram.T,
        shading="auto",
        **(density_kws or {}),
    )
    ax[0].set_title("Mean Histogram")
    ax[0].set_xlabel(x_dimension)
    ax[0].set_ylabel(y_dimension)
    fig.colorbar(c, ax=ax[0], label="Density")

    c = ax[1].pcolormesh(
        x_centers,
        y_centers,
        normalized_confidence_width.T,
        **(ci_kws or {}),
    )
    fig.colorbar(c, ax=ax[1], label="Normalized CI")

    return fig, ax
