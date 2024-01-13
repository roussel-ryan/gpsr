import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.distributions import Uniform, Normal, Exponential
from pyro.infer import MCMC, NUTS, Predictive
import numpy as np
import seaborn as sns
from pyro.infer.autoguide import AutoDelta, AutoNormal

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def mapping(x, offset, width):
    return torch.exp(-(x - offset) ** 2 / width ** 2)


def model(x, data=None):
    offset_mean = pyro.sample(
        "offset_mean", Uniform(-2.0, 2.0)
    )
    offset_var = pyro.sample(
        "offset_var", Uniform(0.0, 2.0)
    )

    sigma = pyro.sample(
        "sigma", Exponential(1.0)
    )
    width = pyro.sample(
        "width", Uniform(0.8, 1.0)
    )

    with pyro.plate("data", len(x)):
        offset = pyro.sample(
            "offset", Normal(offset_mean, offset_var)
        )

        mean = mapping(x, offset, width)

        pyro.sample(
            "obs", Normal(mean, sigma), obs=data
        )


# define data
x = Uniform(torch.tensor(-5.0), torch.tensor(5.0)).rsample([2000])

z = Normal(torch.tensor(-1.0), torch.tensor(1.0)).rsample([len(x)])
print(z)
y = mapping(x, z, 1.0)

# start with MAP estimation
guide = AutoNormal(model)
# set up the optimizer
num_steps = 5000
initial_lr = 0.01
gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
optimizer = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
losses = []
for step in range(num_steps):
    loss = svi.step(x, y)
    losses.append(loss)

    if step % 100 == 0:
        print(f"{step}: {loss}")# : {dict(pyro.get_param_store())}")

# make predictions
predictive = pyro.infer.Predictive(model, guide=guide, num_samples=800)

test_x = torch.linspace(-5, 5, len(x))
svi_samples = predictive(test_x)

# plot posterior samples
fig, ax = plt.subplots(1, 5)
for ele in svi_samples["obs"][:10]:
    ax[0].plot(test_x, ele, alpha=0.25, c="C0")

ax[0].plot(x, y, lw=0, marker="+", markersize=10, c="C1")

offset_samples = svi_samples["offset"].flatten().detach().numpy()

for i, name in enumerate(["offset", "offset_mean", "offset_var", "sigma"]):
    ax[i+1].hist(svi_samples[name].flatten().detach().numpy())
    ax[i+1].set_xlabel(name)

print(np.mean(offset_samples))
print(np.std(offset_samples))
plt.show()
