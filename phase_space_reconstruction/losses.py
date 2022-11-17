import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss


class MENTLoss(Module):
    def __init__(self, lambda_, beta_=torch.tensor(0.0), debug=False):
        super(MENTLoss, self).__init__()

        self.debug = debug
        self.register_parameter("lambda_", Parameter(lambda_))
        self.register_parameter("beta_", Parameter(beta_))

        self.loss_record = []

    def forward(self, outputs, target_image, penalty=0.0):
        pred_image = outputs[0]
        entropy = outputs[1]
        image_loss = mse_loss(pred_image, target_image)
        total_loss = -entropy + self.lambda_ * image_loss + self.beta_ * penalty

        self.loss_record.append([image_loss, entropy, total_loss])

        return total_loss
