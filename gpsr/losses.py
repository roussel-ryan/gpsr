import torch
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss

from gpsr.utils import calculate_centroid, calculate_ellipse


def normalize_images(images):
    """
    Normalizes images tensor so that the
    image pixel intensities add up to 1

    An empty image has no sum to normalize by, so it is divided by 1 instead and
    stays empty. Clamping the divisor at ``finfo.tiny`` would do the same to the
    *values*, but its gradient is 1/tiny ~ 8.5e37 -- a single multiply from
    overflowing float32, which turns a predicted beam that missed the sensor into
    NaN weights. A divisor of 1 keeps that gradient at 1. Images with any
    intensity at all are unaffected, bit for bit.

    Parameters
    ----------
    images: torch.Tensor
        tensor containing images

    Returns
    -------
    normalized images
    """

    sums = images.sum(dim=(-1, -2), keepdim=True)
    return images / torch.where(sums > 0, sums, torch.ones_like(sums))


def kl_div(target, pred):
    eps = 1e-10
    return target * torch.abs((target + eps).log() - (pred + eps).log())


def kl_div_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler divergence ``KL(target || pred)`` as a scalar loss.

    True signed KL, ``sum_i t_i (log t_i - log p_i)``, summed over the image axes
    and averaged over the batch -- unlike :func:`kl_div`, which is per-pixel and
    takes an absolute value.

    KL is defined between probability distributions, so ``target`` and ``pred``
    are expected pre-normalized to unit intensity (see :func:`normalize_images`).
    The target-weighting (each term scales with ``t_i``) is what makes a small
    secondary peak carry weight proportional to its share of the beam's mass while
    the near-zero background contributes ~0 to both loss and gradient. KL is
    asymmetric: it penalizes *missing* target mass but is lenient about predicted
    mass where the target is zero.

    Parameters
    ----------
    target, pred : torch.Tensor
        Image tensors of matching shape ``(..., W, H)``, each normalized so the
        last two axes sum to 1.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence, summed over pixels and averaged over the batch.
    """
    eps = 1e-10
    # target * (log target - log pred), with target as the numerator so background
    # pixels (target ~ 0) vanish rather than blowing up the log ratio. Sum over the
    # image axes gives true KL; mean over the batch keeps the scale resolution- and
    # batch-independent (comparable to an MAE, so lr need not be retuned).
    per_pixel = target * ((target + eps).log() - (pred + eps).log())
    return per_pixel.sum(dim=(-1, -2)).mean()


def center_images_on_centroid(images: torch.Tensor) -> torch.Tensor:
    """Shift each image so its intensity centroid sits at the geometric center.

    Unlike :func:`gpsr.data_processing.center_images`, this takes no centroids --
    it derives each image's own -- and is a differentiable torch op rather than a
    numpy/``scipy.ndimage.shift`` pass, so it can sit inside a training loss.

    Differentiable w.r.t. the pixel intensities: the shift is a per-image integer
    (a cyclic ``roll``, which is just a permutation of pixels, so gradients flow
    straight through). The shift *amount* is rounded from the centroid and carries
    no gradient itself; for a sub-pixel, shift-differentiable centering use a
    Fourier phase shift or ``grid_sample`` instead.

    Wrapping is cyclic (``roll``): intensity pushed off one edge reappears on the
    opposite edge. That is harmless when the beam is compact and away from the
    borders, which is the usual case after centering.

    Parameters
    ----------
    images : torch.Tensor
        Image tensor shaped ``(..., W, H)`` (last axis y/rows, second-to-last
        x/columns -- matching ``gpsr`` conventions). Any number of leading batch dims.

    Returns
    -------
    torch.Tensor
        Images of the same shape, each centered on its own centroid.
    """
    *batch_shape, width, height = images.shape
    flat = images.reshape(-1, width, height)
    n = flat.shape[0]

    # Centroid (in pixel coordinates) of each image, weighted by intensity.
    coord_x = torch.arange(width, device=images.device, dtype=images.dtype)
    coord_y = torch.arange(height, device=images.device, dtype=images.dtype)
    x_proj = flat.sum(dim=-1)  # (n, W)
    y_proj = flat.sum(dim=-2)  # (n, H)
    x_centroid = (x_proj * coord_x).sum(-1) / x_proj.sum(-1).clamp_min(1e-8)
    y_centroid = (y_proj * coord_y).sum(-1) / y_proj.sum(-1).clamp_min(1e-8)

    # Integer shift that moves each centroid to the geometric center. Detached
    # from the graph: the shift is a discrete index, not a differentiable value.
    shift_x = torch.round((width - 1) / 2 - x_centroid).long()  # (n,)
    shift_y = torch.round((height - 1) / 2 - y_centroid).long()  # (n,)

    # Per-image cyclic roll via gather (torch.roll applies one shift to the whole
    # tensor; gather lets each image roll by its own amount). rolled[i] = src[i - s].
    idx_x = (torch.arange(width, device=images.device) - shift_x[:, None]) % width
    flat = flat.gather(1, idx_x[:, :, None].expand(n, width, height))
    idx_y = (torch.arange(height, device=images.device) - shift_y[:, None]) % height
    flat = flat.gather(2, idx_y[:, None, :].expand(n, width, height))

    return flat.reshape(*batch_shape, width, height)


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

        return total_loss * self.gamma_
