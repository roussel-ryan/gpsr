from abc import ABC
import os

import lightning as L
import torch
from torch import optim

from gpsr.losses import mae_loss, normalize_images
from gpsr.modeling import (
    GPSR,
)
from copy import deepcopy
from gpsr.beams import NNParticleBeamGenerator, NNTransform

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class LitGPSR(L.LightningModule, ABC):
    def __init__(self, gpsr_model: GPSR, lr=1e-3):
        super().__init__()
        self.gpsr_model = gpsr_model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # get the training data batch
        x, y = batch

        # make predictions using the GPSR model
        pred = self.gpsr_model(x)

        # check to make sure the prediction shape matches the target shape EXACTLY
        # removing this check will allow the model to run, but it will not be correct
        for i in range(len(pred)):
            if not pred[i].shape == y[i].shape:
                raise RuntimeError(
                    f"prediction {i} shape {pred[i].shape} does not match target shape {y[i].shape}"
                )

        # normalize images
        y_normalized = [normalize_images(y_ele) for y_ele in y]
        pred_normalized = [normalize_images(pred_ele) for pred_ele in pred]

        # add up the loss functions from each prediction (in a tuple)
        diff = [
            mae_loss(y_ele, pred_ele)
            for y_ele, pred_ele in zip(y_normalized, pred_normalized)
        ]

        if len(diff) > 1:
            loss = torch.add(*diff) / len(diff)
        else:
            loss = diff[0]
        # log the loss function at the end of each epoch
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_gpsr(
    model,
    train_dataloader,
    n_epochs=100,
    lr=1e-3,
    logger=None,
    dirpath=None,
    checkpoint_period_epochs=100,
    **kwargs,
):
    """
    Train the GPSR model using PyTorch Lightning.

    Arguments
    ---------
    model: GPSRModel
        GPSR model to be trained.
    train_dataloader: DataLoader
        DataLoader for the training data.
    epochs: int, optional
        Number of epochs to train the model. Default is 100.
    lr: float, optional
        Learning rate for the optimizer. Default is 1e-3.
    log_name: str, optional
        Name of the log file to save training logs. Default is "gpsr".
    checkpoint_period_epochs: int, optional
        Number of epochs between saving checkpoints. Default is 100.
    kwargs: Additional arguments to be passed to the Trainer.

    Returns
    -------
    model: GPSRModel
        Trained GPSR model.

    """

    logger = logger or CSVLogger("logs", name="gpsr")
    dirpath = os.path.join(logger.log_dir, "checkpoints")

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,  # Directory to save checkpoints
        filename="{step}",  # Unique filename
        save_weights_only=False,  # Save the full model (including optimizer state)
        every_n_epochs=checkpoint_period_epochs,
        save_top_k=-1,  # Save all checkpoints
        monitor="loss",  # Monitor the loss for saving checkpoints
    )

    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[periodic_checkpoint_callback],
        **kwargs,
    )

    gpsr_model = LitGPSR(model, lr)
    trainer.fit(
        gpsr_model,
        train_dataloader,
    )

    return model
