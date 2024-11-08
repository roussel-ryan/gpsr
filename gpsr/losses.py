import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss

from phase_space_reconstruction.utils import calculate_centroid, calculate_ellipse


def normalize_images(images):
    """
    Normalizes images tensor so that the
    image pixel intensities add up to 1

    Parameters
    ----------
    images: torch.Tensor
        tensor containing images

    Returns
    -------
    normalized images
    """

    sums = images.sum(dim=(-1, -2), keepdim=True)
    return images / sums


def kl_div(target, pred):
    eps = 1e-10
    return target * torch.abs((target + eps).log() - (pred + eps).log())


def log_mse(target, pred):
    eps = 1e-10
    return mse_loss((target + eps).log(), (pred + eps).log())


def mae_loss(target, pred):
    return torch.mean(torch.abs(target - pred))


def mae_log_loss(target, pred):
    return torch.mean(torch.abs(torch.log(target + 1e-8) - torch.log(pred + 1e-8)))


class MAELoss(Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        
        self.loss_record = []
        
    def forward(self, outputs, target_image_original):
        assert outputs[0].shape == target_image_original.shape
        target_image = normalize_images(target_image_original)
        pred_image = normalize_images(outputs[0])
        
        image_loss = mae_loss(target_image, pred_image)
        
        return image_loss



class MENTLoss(Module):
    def __init__(
        self,
        lambda_,
        beta_=torch.tensor(0.0),
        gamma_=torch.tensor(1.0),
        alpha_=torch.tensor(0.0),
        debug=False,
    ):
        super(MENTLoss, self).__init__()

        self.debug = debug
        self.register_parameter("lambda_", Parameter(lambda_))
        self.register_parameter("beta_", Parameter(beta_))
        self.register_parameter("gamma_", Parameter(gamma_))
        self.register_parameter("alpha_", Parameter(alpha_))

        self.loss_record = []

    def forward(self, outputs, target_image_original):
        assert outputs[0].shape == target_image_original.shape
        target_image = normalize_images(target_image_original)
        # target_image = target_image_original
        pred_image = normalize_images(outputs[0])
        entropy = outputs[1]
        cov = outputs[2]

        # compare image centroids to get regularization
        x = torch.arange(target_image.shape[-1]).to(target_image)
        pred_centroids = calculate_centroid(pred_image, x, x)
        target_centroids = calculate_centroid(target_image, x, x)
        distances = torch.norm(pred_centroids - target_centroids, dim=0)
        centroid_loss = distances.mean() ** 3

        # compare ellipses
        _, pred_covs = calculate_ellipse(pred_image, x, x)
        _, target_covs = calculate_ellipse(target_image, x, x)
        cov_loss = mae_loss(pred_covs, target_covs)

        # image_loss = kl_div(target_image, pred_image).mean()
        image_loss = mae_loss(target_image, pred_image)
        total_loss = (
            -0 * entropy
            + self.lambda_ * image_loss
            + self.beta_ * centroid_loss
            + self.alpha_ * cov_loss
        )

        """
        if 0:
            fig, ax = plt.subplots(4, 2, sharex="all", sharey="all")
            fig.set_size_inches(5, 15)
            ax[0][0].set_title(image_loss.data)
            for i in range(4):
                ax[i][0].imshow(
                    target_image[i][0].cpu().detach(),
                    vmin=0,
                    vmax=0.005,
                    origin="lower",
                )
                ax[i][0].plot(
                    target_centroids[0][i, 0].cpu().detach(),
                    target_centroids[1][i, 0].cpu().detach(),
                    "r+",
                )
                ax[i][1].imshow(
                    pred_image[i][0].cpu().detach(), vmin=0, vmax=0.005, origin="lower"
                )
                ax[i][1].plot(
                    pred_centroids[0][i, 0].cpu().detach(),
                    pred_centroids[1][i, 0].cpu().detach(),
                    "r+",
                )
            print(image_loss)
            plt.show()
        """
        """
        self.loss_record.append(
            [
                torch.tensor(
                    [
                        self.lambda_ * image_loss,
                        -entropy,
                        self.beta_ * centroid_loss,
                        self.alpha_ * cov_loss,
                        total_loss * self.gamma_,
                    ]
                ),
                cov,
            ]
        )
        """
        return total_loss * self.gamma_
