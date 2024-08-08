from abc import ABC

import lightning as L
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
        x, y = batch

        pred, beam = self.gpsr_model(x)
        loss = mae_loss(y, pred)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


