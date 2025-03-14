from abc import ABC

import lightning as L
import torch
from torch import optim

from gpsr.losses import mae_loss
from gpsr.modeling import (
    GPSR,
)


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

        # add up the loss functions from each prediction (in a tuple)
        diff = [mae_loss(y_ele, pred_ele) for y_ele, pred_ele in zip(y, pred)]

        if len(diff) > 1:
            loss = torch.add(*diff)
        else:
            loss = diff[0]
        # log the loss function at the end of each epoch
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
