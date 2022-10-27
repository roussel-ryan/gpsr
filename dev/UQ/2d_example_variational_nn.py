import matplotlib.pyplot as plt
import torch

from models import IndependentVariationalNN
from phase_space_reconstruction.histogram import histogram, histogram2d


def gt_dist(x):
    prob = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2) * 0.1
    ).log_prob(x).exp()
    return prob


def pred_probability(x, y):
    bandwidth = torch.tensor(1e-1)
    bin_width = x[1] - x[0]
    hist = histogram2d(*torch.movedim(y, -1, 0), x, bandwidth).squeeze()
    return hist / bin_width ** 2


# probability of data given y
def log_predictive_likelihood(x, y, target_proj, visualize=False):
    # y is tensor of particle coordiantes

    # calculate KDE of y projected onto the x axis
    pred_p = pred_probability(x, y).sum(dim=-1)

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(x, pred_p.detach()[0])
        ax.plot(x, target_proj.detach())
        # ax.plot(x, (log_target_p - log_pred_p)[0].detach())

        # fig2,ax2 = plt.subplots()
        # ax2.plot(x,(log_target_p.exp()*(log_target_p - log_pred_p))[0].detach())

    # calc kl div between distributions
    # return (log_target_p.exp() * (log_target_p - log_pred_p)).sum(dim=-1)
    return torch.sum((target_proj - pred_p) ** 2, dim=-1)


n_mesh = 25
x = torch.linspace(-1.5, 1.5, n_mesh)
mesh_x = torch.meshgrid(x, x)
test_x = torch.stack(mesh_x, dim=-1)

# log target projection
gt_val = gt_dist(test_x)
gt_proj = torch.sum(gt_val, dim=-1)

n_part = 5000
n_samples = 3
model = IndependentVariationalNN(n_outputs=2)
base_X = torch.rand(n_part, 2).repeat(n_samples, 1, 1) * 2.0 - 1.0
# model = precondition(model, base_X)

out = model(base_X)
fig, ax = plt.subplots()
ax.hist2d(*out[0].T.detach().numpy(), bins=20)

# train
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_track = []
for i in range(3000):
    optim.zero_grad()

    # get samples
    out = model(base_X).squeeze()
    log_lk = log_predictive_likelihood(x, out, gt_proj, visualize=False).mean()
    complexity = model.std(base_X).mean() * 5.0

    # calculate loss
    loss = log_lk - complexity

    loss.backward()
    loss_track += [torch.stack([loss, log_lk, complexity])]

    if i % 100 == 0:
        print(log_lk, complexity, loss)

    optim.step()

log_predictive_likelihood(x, out, gt_proj, visualize=False).mean()

loss_track = torch.stack(loss_track).detach()
fig2, ax2 = plt.subplots()
ax2.plot(loss_track)

out = model(base_X)
fig, ax = plt.subplots()

bandwidth = torch.tensor(1e-1)
for i in range(len(out)):
    ax.plot(x, histogram(out[i, :, 1].detach(), x, bandwidth=bandwidth))

#ax.hist2d(*out[0].T.detach().numpy(), bins=20)

plt.show()
