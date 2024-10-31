from abc import ABC

import lightning as L
import torch
from torch import optim

from phase_space_reconstruction.losses import mae_loss
from phase_space_reconstruction.modeling import (
    GPSR,
)


class LitGPSR(L.LightningModule, ABC):
    def __init__(self, gpsr_model: GPSR):
        super().__init__()
        self.gpsr_model = gpsr_model

    def training_step(self, batch, batch_idx):
        # get the training data batch
        x, y = batch

        # make predictions using the GPSR model
        pred = self.gpsr_model(x)

        # add up the loss functions from each prediction (in a tuple)
        loss = torch.add(
            *[mae_loss(y_ele, pred_ele) for y_ele, pred_ele in zip(y, pred)]
        )
        # log the loss function at the end of each epoch
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
