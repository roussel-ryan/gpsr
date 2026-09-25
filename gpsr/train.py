import os
from abc import ABC
from typing import Callable
from torch.utils.data import DataLoader

from gpsr.beams import ResNNTransform
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
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
    train_dataloader: DataLoader,
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
    logger: optional
        Logger passed to the Lightning `Trainer`. Default is a new `CSVLogger` with name "gpsr".
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


def train_gpsr_multistep(
    gpsr_model: GPSR,
    train_dataloader: DataLoader,
    n_epochs_linear: int = 100,
    n_epochs_full: int = 100,
    lr_linear: float = 1e-3,
    lr_full: float = 1e-3,
    loss_func: Callable = mae_loss,
    logger=None,
    dirpath=None,
    checkpoint_period_epochs: int = 100,
    **kwargs,
):
    """
    Train a GPSR model that uses an instance of ResNNTransform as the transformer in two stages.

    In the first stage, only the linear parameters of the beam generator's
    transformer (`transformer.linear_parameters`, e.g. `ResNNTransform`'s first
    layer) are trained, with all other parameters frozen. If the transformer has
    a trainable `alpha` (skip-connection scale) parameter, it is also zeroed out
    for this stage so that the model's output is purely linear. 
    
    In the second stage, the linear parameters are frozen and the remaining parameters are
    trained instead. This lets the linear transformation fit the coarse, dominant
    behavior of the beam before the (harder to fit) nonlinear residual
    parameters are trained, which can improve training stability/convergence.

    Arguments
    ---------
    gpsr_model: GPSR
        GPSR model to be trained that has a transformer of type `ResNNTransform`.
    train_dataloader: DataLoader
        DataLoader for the training data.
    n_epochs_linear: int, optional
        Number of epochs for the linear-only training stage. Default is 100.
    n_epochs_full: int, optional
        Number of epochs for the full-model training stage. Default is 100.
    lr_linear: float, optional
        Learning rate used during the linear-only stage. Default is 1e-3.
    lr_full: float, optional
        Learning rate used during the full-model stage. Default is 1e-3.
    loss_func: Callable, optional
        Loss function used in both stages. Default is `mae_loss`.
    logger: optional
        Logger passed to the Lightning `Trainer` in both stages. Default is a new
        `CSVLogger` per stage.
    checkpoint_period_epochs: int, optional
        Number of epochs between saving checkpoints. Default is 100.
    kwargs: Additional arguments to be passed to the Trainer in both stages.

    Returns
    -------
    lit_gpsr_model: LitGPSR
        Trained LitGPSR model after both stages.

    """
    transformer = gpsr_model.beam_generator.transformer
    if not isinstance(transformer, ResNNTransform):
        raise TypeError(
            f"Expected transformer to be an instance of ResNNTransform, but got {type(transformer).__name__}"
        )
    
    linear_params = list(transformer.linear_parameters)
    linear_param_ids = {id(p) for p in linear_params}
    non_linear_params = [
        p for p in gpsr_model.parameters() if id(p) not in linear_param_ids
    ]

    # stage 1: freeze everything except the transformer's linear parameters
    for p in non_linear_params:
        p.requires_grad_(False)

    # zero out the skip-connection scale so stage 1's output is purely linear
    alpha = getattr(transformer, "alpha", None)
    if isinstance(alpha, torch.nn.Parameter):
        with torch.no_grad():
            alpha.zero_()

    lit_gpsr_model = train_gpsr(
        gpsr_model,
        train_dataloader,
        n_epochs=n_epochs_linear,
        lr=lr_linear,
        loss_func=loss_func,
        logger=logger,
        dirpath=dirpath,
        checkpoint_period_epochs=checkpoint_period_epochs,
        **kwargs,
    )

    # stage 2: freeze the linear parameters and train everything else
    for p in non_linear_params:
        p.requires_grad_(True)
    for p in linear_params:
        p.requires_grad_(False)

    lit_gpsr_model = train_gpsr(
        lit_gpsr_model.gpsr_model,
        train_dataloader,
        n_epochs=n_epochs_full,
        lr=lr_full,
        loss_func=loss_func,
        logger=logger,
        dirpath=dirpath,
        checkpoint_period_epochs=checkpoint_period_epochs,
        **kwargs,
    )

    # restore the linear parameters to trainable so the returned model is unfrozen
    for p in linear_params:
        p.requires_grad_(True)

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


class EpochLossPrinter(Callback):
    """Print the epoch loss every N epochs."""

    def __init__(self, every_n_epochs: int = 100, metric: str = "loss"):
        self.every_n_epochs = every_n_epochs
        self.metric = metric

    def on_train_epoch_end(self, trainer, pl_module):
        is_last = trainer.current_epoch == trainer.max_epochs - 1
        if trainer.current_epoch % self.every_n_epochs and not is_last:
            return

        loss = trainer.callback_metrics.get(self.metric)
        loss = "N/A" if loss is None else f"{float(loss):.2e}"
        print(
            f"epoch {trainer.current_epoch:>5}/{trainer.max_epochs} | {self.metric}={loss}"
        )
