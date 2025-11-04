import os
from abc import ABC
from typing import Callable

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import optim

from gpsr.losses import mae_loss, normalize_images
from gpsr.modeling import GPSR
from gpsr.modeling import EntropyGPSR


class LitGPSR(L.LightningModule, ABC):
    def __init__(
        self, gpsr_model: GPSR, lr: float = 1e-3, loss_func: Callable = mae_loss
    ):
        super().__init__()
        self.gpsr_model = gpsr_model
        self.lr = lr
        self.loss_func = loss_func

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

        # add up the loss functions from each prediction
        loss = 0.0
        for y_ele, pred_ele in zip(y, pred):
            # normalize images
            y_ele_norm = normalize_images(y_ele)
            pred_ele_norm = normalize_images(pred_ele)
            # add loss
            loss += self.loss_func(y_ele_norm, pred_ele_norm)

        # normalize loss by number of outputs
        loss /= len(y)

        # log the loss function at the end of each epoch
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_gpsr(
    gpsr_model: GPSR,
    train_dataloader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    loss_func: Callable = mae_loss,
    logger=None,
    dirpath=None,
    checkpoint_period_epochs: int = 100,
    **kwargs,
):
    """
    Train the GPSR model using PyTorch Lightning.

    Arguments
    ---------
    gpsr_model: GPSR
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
    lit_gpsr_model: LitGPSR
        Trained LitGPSR model.

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

    lit_gpsr_model = LitGPSR(gpsr_model, lr, loss_func=loss_func)
    trainer.fit(
        lit_gpsr_model,
        train_dataloader,
    )

    return lit_gpsr_model


class EntropyLitGPSR(L.LightningModule, ABC):
    """Minimizes entropy-regularized loss function.

    L = H + lambda * D, where H is the entropy and D is the prediction error, and
    lambda is a constant (penalty parameter).
    """

    def __init__(
        self, gpsr_model: EntropyGPSR, lr: float = 0.001, penalty: float = 0.0
    ) -> None:
        super().__init__()
        self.gpsr_model = gpsr_model
        self.lr = lr
        self.penalty = penalty

    def training_step(self, batch, batch_idx):
        # get the training data batch
        x, y = batch

        # make predictions using the GPSR model
        (beam, entropy, pred) = self.gpsr_model(x)

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

        loss_pred = 0.0
        if len(diff) > 1:
            loss_pred += torch.add(*diff) / len(diff)
        else:
            loss_pred += diff[0]

        # compute regularization term (negative entropy)
        loss_reg = -entropy
        loss = loss_reg + self.penalty * loss_pred

        self.log("loss_pred", loss_pred, on_epoch=True)
        self.log("loss_reg", loss_reg, on_epoch=True)
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpsr_model.parameters(), lr=self.lr)
        return optimizer
