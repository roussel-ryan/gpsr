import torch
from torch.nn import Parameter

from utils import kl_div


class WeightedConstrainedLoss(torch.nn.MSELoss):
    def __init__(self, l0):
        super().__init__()
        self.loss_record = []
        self.register_parameter("lambda_", Parameter(l0))

    def forward(self, input_data, target):
        image_loss = kl_div(target, input_data[0]).sum()
        entropy_loss = -input_data[1]
        constrained_loss = entropy_loss + self.lambda_ * image_loss

        self.loss_record.append(
            [image_loss, entropy_loss, input_data[2], self.lambda_.data]
        )

        return constrained_loss


class GradientSquaredLoss(torch.nn.MSELoss):
    def __init__(self, l0, model):
        super().__init__()
        self.loss_record = []
        self.register_parameter("lambda_", Parameter(l0))
        #self.model = model

    def forward(self, input_data, target):
        image_loss = kl_div(target, input_data[0]).sum()
        entropy_loss = -input_data[1]
        # print(entropy_loss.data, image_loss.data, self.lambda_.data)

        # attempt to maximize the entropy loss while constraining on the image loss
        # using lagrange multipliers see:
        # https://en.wikipedia.org/wiki/Lagrange_multiplier

        unconstrained_loss = entropy_loss + self.lambda_ * image_loss
        #z = grad(
        #    unconstrained_loss,
        #    list(self.model.beam_generator.parameters()) + [self.lambda_],
        #    create_graph=True,
        #)
        #grad_loss = torch.norm(
        #    torch.cat([ele.flatten().unsqueeze(1) for ele in z], dim=0)
        #)
        self.loss_record.append(
            [image_loss, entropy_loss, input_data[2], self.lambda_.data]
        )

        return unconstrained_loss