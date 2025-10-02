import torch
from pprint import pprint
import yaml

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gpsr.modeling import GPSR
from gpsr.beams import NNTransform, NNParticleBeamGenerator
from gpsr.train import LitGPSR

import os
import re


class GPSRRun:
    """
    A class to manage the setup and execution of a GPSR training run.
    This includes preparing datasets, models, logging, checkpointing, and trainer setup.
    """

    def __init__(
        self,
        gpsr_lattice,
        log_name="scans",
        train_dataset=None,
        N_particles=int(5e4),
        n_hidden=2,
        hidden_width=20,
        output_scale=1e-4,
        dropout=0.0,
        batch_size=100,
        max_epochs=5000,
        p0c=1000 * 1e6,
        learning_rate=10e-3,
        checkpoint_period_epochs=100,
        **extra_hparams,
    ):
        """
        Initializes the GPSRRun object with model hyperparameters and training settings.

        Args:
            gpsr_lattice: The beamline or lattice structure for the GPSR model.
            log_name (str): Name of the directory where logs will be saved.
            train_dataset (Dataset, optional): Dataset to be used for training.
            N_particles (int): Number of particles in the simulation.
            n_hidden (int): Number of hidden layers in the NNTransform.
            hidden_width (int): Width of each hidden layer.
            output_scale (float): Scaling factor for NNTransform outputs.
            dropout (float): Dropout rate for the model.
            batch_size (int): Batch size for training.
            max_epochs (int): Number of epochs to train.
            p0c (float): Reference momentum of the beam (in eV/c).
            learning_rate (float): Learning rate for training.
            checkpoint_period_epochs (int): Interval (in epochs) to save checkpoints.
            extra_hparams (dict): Any additional hyperparameters.
        """
        self.gpsr_lattice = gpsr_lattice
        self.hparams = {
            "N_particles": N_particles,
            "n_hidden": n_hidden,
            "hidden_width": hidden_width,
            "output_scale": output_scale,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "p0c": p0c,
            "learning_rate": learning_rate,
            "checkpoint_period_epochs": checkpoint_period_epochs,
        } | extra_hparams

        self.log_name = log_name
        self.train_dataset = train_dataset

        print("Hyperparameters:")
        pprint(self.hparams)

    def setup_training(self):
        """
        Setup the full training pipeline, including dataset, model, logger, checkpointing, and trainer.

        """

        # Initialize the GPSR model with the lattice and particle generator
        self.gpsr_model = self.setup_gpsr_model()

        # Wrap the GPSR model in the LitGPSR Lightning module
        self.litgpsr = self.setup_litgpsr()

        # Prepare the DataLoader for training
        self.train_loader = self.setup_trainloader()

        # Setup logger for tracking metrics and checkpoints
        self.logger = self.setup_logger()
        self.logger.log_hyperparams(self.hparams)

        # Setup checkpointing to save model progress
        self.checkpoint_callback = self.setup_checkpointing()

        # Setup PyTorch Lightning Trainer
        self.trainer = self.setup_trainer()

    def train(self):
        """
        Start the training process using the configured trainer and DataLoader.
        """
        print(f"Running training - results will be saved in {self.logger.log_dir}")
        self.trainer.fit(self.litgpsr, self.train_loader)

    def setup_gpsr_model(self):
        """
        Initialize the GPSR model using provided hyperparameters and lattice.

        Returns:
            GPSR: The initialized GPSR model object.
        """
        return GPSR(
            NNParticleBeamGenerator(
                self.hparams["N_particles"],
                self.hparams["p0c"],
                transformer=NNTransform(
                    self.hparams["n_hidden"],
                    self.hparams["hidden_width"],
                    output_scale=self.hparams["output_scale"],
                ),
            ),
            self.gpsr_lattice,
        )

    def setup_litgpsr(self):
        """
        Wrap the GPSR model in the LitGPSR Lightning module.

        Returns:
            LitGPSR: The wrapped model ready for training.
        """
        return LitGPSR(self.gpsr_model, self.hparams["learning_rate"])

    def setup_trainloader(self):
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: The PyTorch DataLoader object.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams["batch_size"]
        )

    def setup_logger(self):
        """
        Setup the CSV logger for experiment tracking.

        Returns:
            CSVLogger: The logger object.
        """
        return CSVLogger("logs", name=self.log_name)

    def setup_checkpointing(self):
        """
        Configure model checkpointing to save progress at specified intervals.

        Returns:
            ModelCheckpoint: The checkpoint callback.
        """
        dirpath = os.path.join(self.logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,  # Directory to save checkpoints
            filename="{step}",  # Filename pattern
            save_weights_only=False,  # Save full model (not just weights)
            every_n_epochs=self.hparams["checkpoint_period_epochs"],
            save_top_k=-1,  # Save all checkpoints
            monitor="loss",  # Monitor loss for saving
        )
        return checkpoint_callback

    def setup_trainer(self):
        """
        Setup the PyTorch Lightning Trainer with max epochs, logger, and callbacks.

        Returns:
            Trainer: The PyTorch Lightning Trainer.
        """
        return L.Trainer(
            max_epochs=self.hparams["max_epochs"],
            logger=self.logger,
            callbacks=[self.checkpoint_callback],
        )

    @classmethod
    def from_checkpoint(
        cls, gpsr_lattice, log_name, version_no, checkpoint_number=-1, extra_hparams={}
    ):
        """
        Load a GPSRRun instance from a saved checkpoint.

        Args:
            log_name (str): Name of the log directory.
            version_no (int): Version number of the experiment.
            checkpoint_number (int): Index of the checkpoint to load (-1 for the latest).
            extra_hparams (dict): Extra hyperparameters to override.

        Returns:
            GPSRRun: The loaded GPSRRun instance.
        """
        # Load hyperparameters from saved YAML
        with open(f"{log_name}/version_{version_no}/hparams.yaml") as stream:
            hparams = yaml.safe_load(stream)

        # Initialize the run
        run = cls(gpsr_lattice, **hparams, log_name=log_name)
        run.hparams.update(extra_hparams)

        # Re-setup model components
        run.gpsr_model = run.setup_gpsr_model()

        # Get checkpoint filename
        checkpoint_file_name = run.list_checkpoint_filenames(version_no)[
            checkpoint_number
        ]

        print(f"Loading checkpoint {checkpoint_file_name}...")

        # Load the Lightning module from checkpoint
        run.litgpsr = LitGPSR.load_from_checkpoint(
            f"{checkpoint_file_name}",
            gpsr_model=run.gpsr_model,
            strict=False,
            map_location=torch.device("cpu"),
        )

        return run

    def list_checkpoint_filenames(self, version_no):
        """
        List all checkpoint filenames for a given version, sorted by step number.

        Args:
            version_no (int): Version number of the experiment.

        Returns:
            list: Sorted list of checkpoint file paths.
        """
        checkpoint_filenames = [
            f"{self.log_name}/version_{version_no}/checkpoints/" + name
            for name in sorted(
                os.listdir(f"{self.log_name}/version_{version_no}/checkpoints")
            )
        ]

        # Helper function to extract step number from filename
        def extract_epoch(filename):
            match = re.search(r"step=(\d+).", filename)
            step = int(match.group(1)) if match else float("inf")
            return step

        # Sort filenames by extracted step number
        sorted_filenames = sorted(checkpoint_filenames, key=extract_epoch)

        return sorted_filenames
