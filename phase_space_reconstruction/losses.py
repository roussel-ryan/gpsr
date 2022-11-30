import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss


def kl_div(target, pred):
    eps = 1e-10
    return target * torch.abs((target + eps).log() - (pred + eps).log())


class MENTLoss(Module):
    def __init__(self, lambda_, beta_=torch.tensor(0.0), gamma_=torch.tensor(1.0), debug=False):
        super(MENTLoss, self).__init__()

        self.debug = debug
        self.register_parameter("lambda_", Parameter(lambda_))
        self.register_parameter("beta_", Parameter(beta_))
        self.register_parameter("gamma_", Parameter(gamma_))

        self.loss_record = []

    def forward(self, outputs, target_image, penalty=0.0):
        pred_image = outputs[0]
        entropy = outputs[1]
        #image_loss = kl_div(target_image, pred_image).mean()
        image_loss = mse_loss(target_image, pred_image)
        total_loss = -entropy + self.lambda_ * image_loss + self.beta_ * penalty

        if 0:
            fig, ax = plt.subplots(4, 2, sharex="all", sharey="all")
            fig.set_size_inches(5, 15)
            ax[0][0].set_title(image_loss.data)
            for i in range(4):
                ax[i][0].imshow(target_image[i][0].cpu().detach(), vmin=0, vmax=0.005)
                ax[i][1].imshow(pred_image[i][0].cpu().detach(), vmin=0, vmax=0.005)

            plt.show()

        self.loss_record.append(
            torch.tensor([self.lambda_ * image_loss, entropy, total_loss]))

        return total_loss * self.gamma_
