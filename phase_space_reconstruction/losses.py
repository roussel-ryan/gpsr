from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss


class MENTLoss(Module):
    def __init__(self, lambda_):
        super(MENTLoss, self).__init__()

        self.register_parameter("lambda_", Parameter(lambda_))

    def forward(self, pred_image, target_image, entropy):
        image_loss = mse_loss(pred_image, target_image)
        return -entropy + self.lambda_ * image_loss
