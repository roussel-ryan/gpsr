import matplotlib.pyplot as plt
import torch

from models import CoupledVariationalNN, precondition
from phase_space_reconstruction.histogram import histogram


def log_target_probability(x):
    """returns the log of the target probability distribution"""

    bin_width = x[1] - x[0]
    out = (
            torch.distributions.Normal(torch.ones(1) * -0.5, 0.2 * torch.ones(1))
            .log_prob(x)
            .exp()
            + torch.distributions.Normal(torch.ones(1) * 0.5, 0.2 * torch.ones(1))
            .log_prob(x)
            .exp()
    )
    out = (out / torch.sum(out)) / bin_width
    return torch.log(out)


def log_pred_probability(x, y):
    """
    calculates a differentiable log-probability distribution at points `x` from
    samples `y`

    """
    bandwidth = torch.tensor(1e-1)
    bin_width = x[1] - x[0]
    hist = histogram(y, x, bandwidth).squeeze()
    return torch.log(hist / bin_width + 1e-8)


# probability of data given y
def log_predictive_likelihood(x, y, visualize=False):
    # y is tensor of particle coordiantes

    # target distribution
    log_target_p = log_target_probability(x)

    # calculate KDE of y
    log_pred_p = log_pred_probability(x, y)

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(x, log_pred_p.detach()[0].exp())
        ax.plot(x, log_target_p.detach().exp())
        # ax.plot(x, (log_target_p - log_pred_p)[0].detach())

        # fig2,ax2 = plt.subplots()
        # ax2.plot(x,(log_target_p.exp()*(log_target_p - log_pred_p))[0].detach())

    # calc kl div between distributions
    return (
            torch.sum(torch.abs(log_target_p - log_pred_p), dim=-1) /
            log_target_p.shape[-1]
    )


test_x = torch.linspace(-1.5, 1.5, 100)
n_part = 500

n_samples = 50
model = CoupledVariationalNN()
# base_X = torch.rand(n_part, 1).repeat(n_samples, 1, 1)*2.0 - 1.0

# precondition model weights to predict a uniform distribution
base_X = torch.linspace(-0.1, 0.1, n_part).unsqueeze(-1).repeat(n_samples, 1, 1)
model = precondition(model, base_X)

# show initial distribution
out = model(base_X).squeeze()
fig, ax = plt.subplots()
ax.hist(base_X[0].detach().numpy())
fig, ax = plt.subplots()
for ele in out:
    ax.hist(ele.detach().numpy(), histtype="step")

log_predictive_likelihood(test_x, out, True)

# train
optim = torch.optim.Adam(model.parameters(), lr=0.01)

loss_track = []
for i in range(2000):
    optim.zero_grad()

    # get samples
    out = model(base_X).squeeze()
    log_lk = log_predictive_likelihood(test_x, out, visualize=False).mean()
    complexity = model.std(base_X).mean() * 0.1

    # calculate loss
    loss = log_lk - complexity

    loss.backward()
    loss_track += [torch.stack([loss, log_lk, complexity])]

    if i % 100 == 0:
        print(i, log_lk, complexity, loss)

    optim.step()

loss_track = torch.stack(loss_track).detach()
fig2, ax2 = plt.subplots()
ax2.plot(loss_track, label=["total loss", "log lk", "complexity"])
ax2.legend()

# show final distribution
fig, ax = plt.subplots()
out = model(base_X).squeeze()
mean = model.mean(base_X).squeeze()[0].detach()
std = model.std(base_X).squeeze()[0].detach()

initial_dist = log_pred_probability(test_x, out).exp().detach()
initial_dist_mean = initial_dist.mean(dim=0)
l = initial_dist.quantile(0.05, dim=0)
u = initial_dist.quantile(0.95, dim=0)


for i in range(base_X.shape[0]):
    ax.plot(test_x, initial_dist[i].detach())

log_target_p = log_target_probability(test_x)

fig, ax = plt.subplots()
ax.plot(test_x, log_target_p.detach().exp())
ax.plot(test_x, initial_dist_mean)
ax.fill_between(test_x, l, u, alpha=0.5)

fig3, ax3 = plt.subplots()
for i in range(len(mean))[::10]:
    ax3.plot(test_x, torch.distributions.Normal(mean[i], std[i]).log_prob(test_x).exp())

#
# # visualize
# fig2, ax2 = plt.subplots()
# total_preds = []
# for _ in range(100):
#     total_preds += [log_pred_probability(mesh_x, particle_coords.sample()).exp()]
#
# total_preds = torch.stack(total_preds).detach()
# mean = torch.mean(total_preds, dim=0)
# l = torch.quantile(total_preds, 0.05, dim=0)
# u = torch.quantile(total_preds, 0.95, dim=0)
#
# # ax2.plot(mesh_x, total_preds.T.detach(),"r")
# ax2.plot(mesh_x, mean)
# ax2.fill_between(mesh_x, l, u, alpha=0.25)
#
# mean_particle_coords = particle_coords.mu.detach()
# std_particle_coords = torch.log1p(particle_coords.rho.exp()).detach()
#
# fig, ax = plt.subplots()
# for i in range(len(mean_particle_coords)):
#     ax.plot(
#         mesh_x,
#         torch.distributions.Normal(mean_particle_coords[i], std_particle_coords[i])
#         .log_prob(mesh_x)
#         .exp(),
#     )

plt.show()
