import matplotlib.pyplot as plt
import torch
from blitz.modules.weight_sampler import TrainableRandomDistribution
from phase_space_reconstruction.histogram import histogram
from torch.nn.functional import kl_div, mse_loss


def log_target_probability(X):
    return torch.distributions.Normal(torch.ones(1), 0.1 * torch.ones(1)).log_prob(X)


def log_pred_probability(x, y):
    bandwidth = torch.tensor(1e-1)
    bin_width = x[1] - x[0]
    return torch.log(
        histogram(y.reshape(1, -1), x, bandwidth).squeeze() / bin_width + 1e-8
    )


# probability of data given y
def log_predictive_likelihood(x, y, visualize=False):
    # y is tensor of particle coordiantes

    # target distribution
    log_target_p = log_target_probability(x)

    # calculate KDE of y
    log_pred_p = log_pred_probability(x, y)

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(x, log_pred_p.exp().detach())
        ax.plot(x, log_target_p.exp())

    # calc rms difference between densities
    return -(kl_div(log_target_p, log_pred_p, log_target=True)).log()


mesh_x = torch.linspace(0, 2, 100)

n_part = 100
mu_0 = torch.zeros(n_part)
rho_0 = torch.zeros(n_part)
particle_coords = TrainableRandomDistribution(mu_0, rho_0)

# train
optim = torch.optim.Adam(particle_coords.parameters(), lr=0.1)

loss_track = []
n_samples = 3
for i in range(3000):
    optim.zero_grad()

    # average loss over multiple samples
    total_loss = 0
    for _ in range(n_samples):
        sample = particle_coords.sample()

        # calculate loss function (assuming constant prior over locations)
        log_pl = log_predictive_likelihood(mesh_x, sample)
        log_variational_complexity = particle_coords.log_posterior() / n_part

        total_loss += log_variational_complexity - log_pl
    mean_loss = total_loss / n_samples
    mean_loss.backward()
    loss_track += [torch.clone(mean_loss).detach()]

    if i % 100 == 0:
        print(mean_loss)

    optim.step()

fig, ax = plt.subplots()
ax.plot(loss_track)

# visualize
fig2, ax2 = plt.subplots()
total_preds = []
for _ in range(100):
    total_preds += [log_pred_probability(mesh_x, particle_coords.sample()).exp()]

total_preds = torch.stack(total_preds).detach()
mean = torch.mean(total_preds, dim=0)
l = torch.quantile(total_preds, 0.05, dim=0)
u = torch.quantile(total_preds, 0.95, dim=0)

# ax2.plot(mesh_x, total_preds.T.detach(),"r")
ax2.plot(mesh_x, mean)
ax2.fill_between(mesh_x, l, u, alpha=0.25)

mean_particle_coords = particle_coords.mu.detach()
std_particle_coords = torch.log1p(particle_coords.rho.exp()).detach()

fig, ax = plt.subplots()
for i in range(len(mean_particle_coords)):
    ax.plot(
        mesh_x,
        torch.distributions.Normal(mean_particle_coords[i], std_particle_coords[i])
        .log_prob(mesh_x)
        .exp(),
    )

plt.show()
