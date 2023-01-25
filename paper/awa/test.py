import matplotlib.pyplot as plt
from fitting_uncertainty import load_data
import torch
from phase_space_reconstruction.utils import calculate_ellipse

labels = [
    "x (mm)",
    r"$p_x$ (mrad)",
    "y (mm)",
    r"$p_y$ (mrad)",
    "z (mm)",
    r"$p_z$ (mrad)",
]

tkwargs = {"dtype": torch.float}
save_dir = "uncertainty/ensemble_mse_scale_0.95"
quad_strengths, image_data, bins, xx = load_data(tkwargs)


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
def confidence_ellipse(mean, cov, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse

covs = []
for i in range(image_data.shape[0]):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 30)
    image = image_data[i].cpu().detach()
    center, cov = calculate_ellipse(image, bins, bins)
    #ellipse = confidence_ellipse(center.numpy(), cov.numpy(), edgecolor='red')

    #ax.pcolor(*xx, image, vmin=0, vmax=0.01)
    #ax.set_title(cov.flatten().sqrt())
    #ax.add_patch(ellipse)

    covs += [cov]

covs = torch.stack(cov)
torch.save("image_cov.pt")

plt.show()
