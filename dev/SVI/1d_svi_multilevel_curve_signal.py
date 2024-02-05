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


pyro.clear_param_store()

def mapping(x, offset, width):
    return torch.exp(-(x - offset) ** 2 / width ** 2)


# create a generative model that produces signals with random offsets
def model(x, data=None):
    offset_mean = pyro.sample(
        "offset_mean", Uniform(-2.0, 2.0)
    )
    offset_var = pyro.sample(
        "offset_var", Uniform(0.0, 2.0)
    )
    sigma = pyro.sample(
        "sigma", Exponential(10.0)
    )

    d_size = 10 if data is None else data.shape[-1]

    with pyro.plate("shots", d_size):
        offset = pyro.sample(
            "offset", Normal(offset_mean, offset_var)
        )
        with pyro.plate("bins", len(x)):
            mean = mapping(x.unsqueeze(-1), offset, torch.tensor(1.0))

            return pyro.sample(
                "obs", Normal(mean, sigma), obs=data
            )


# test generator
test_x = torch.linspace(-2, 2, 100)
samples = model(test_x).T

fig, ax = plt.subplots()
for ele in samples:
    ax.plot(test_x, ele, c="C0", alpha=0.25)


# create data samples -- corrupt with noise
z = Normal(torch.tensor(-1.0), torch.tensor(0.75)).rsample([10])
y = mapping(test_x.unsqueeze(1), z, 1.0)
y += torch.randn(y.shape)*0.01
for ele in y.T:
    ax.plot(test_x, ele, c="C1")

#plt.show()

pyro.render_model(model, model_args=(test_x, y), filename="model.png")

# do SVI
guide = AutoNormal(model)
num_steps = 1000
initial_lr = 0.01
gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
optimizer = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
losses = []
for step in range(num_steps):
    loss = svi.step(test_x, y)
    losses.append(loss)

    if step % 100 == 0:
        print(f"{step}: {loss}")# : {dict(pyro.get_param_store())}")

# make predictions
predictive = pyro.infer.Predictive(model, guide=guide, num_samples=800)
svi_samples = predictive(test_x, data=None)

# plot posterior samples
params = ["offset", "offset_mean", "offset_var", "sigma"]
fig, ax = plt.subplots(1, len(params) + 1)
for i, name in enumerate(params):
    ax[i].hist(svi_samples[name].flatten().detach().numpy())
    ax[i].set_xlabel(name)

for sample in svi_samples["obs"][:20]:
    for i in range(sample.shape[-1]):
        ax[-1].plot(test_x, sample[..., i], alpha=0.25, c="C0")

for ele in y.T:
    ax[-1].plot(test_x, ele, c="C1")
ax[-1].set_xlabel("x")
ax[-1].set_ylabel("y")

plt.show()
