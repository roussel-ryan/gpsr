import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.distributions import Uniform, Normal, Gamma, Exponential
from pyro.infer import MCMC, NUTS, Predictive
import numpy as np
import seaborn as sns


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
        "sigma", Uniform(0.0, 1.0)
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
x = Uniform(torch.tensor(-2.0), torch.tensor(2.0)).rsample([1])

z = Normal(torch.tensor(-1.0), torch.tensor(0.4)).rsample([len(x)])
print(z)
y = mapping(x, z, 1.0)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
mcmc.run(x, y)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

# plot posterior
fig, ax = plt.subplots(1, 2)

# plot predictive
pred = Predictive(model, hmc_samples)
test_x = x  # torch.linspace(-2.0, 2.0, 10)
results = pred(test_x)


posterior_function_samples = mapping(
        test_x,
        hmc_samples["offset"],
        np.expand_dims(hmc_samples["width"], 1)
)

for ele in posterior_function_samples:
    ax[0].plot(test_x, ele, alpha=0.25, c="C0")

#for ele in results["obs"][:30]:
#    ax[1].plot(test_x, ele, alpha=0.25, c="C1")

ax[0].plot(x, y, lw=0, marker="+", markersize=10, c="C1")

ax[1].hist(hmc_samples["offset"].flatten())
#sns.pairplot(pd.DataFrame(hmc_samples).iloc[:100])
print(np.mean(hmc_samples["offset"]))
print(np.std(hmc_samples["offset"]))

plt.show()
