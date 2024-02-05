# demonstrate pyro model with some stochastic elements
import torch
from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.infer import SVI, Trace_ELBO
import pyro


class BayesianRegression(PyroModule):
    def __init__(self):
        super().__init__()
        self.linear = PyroModule[nn.Linear](1, 1)
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([1,1]).to_event(2)
        )
        #self.linear.bias = PyroSample(
        #    dist.Normal(0.0, 1.0).expand([1]).to_event(1)
        #)

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 0.1))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


x = torch.linspace(0, 1, 10).unsqueeze(1)
y = (x + torch.randn(x.shape) * 0.01).flatten()
print(x.shape)
print(y.shape)

model = BayesianRegression()
guide = AutoNormal(model)

adam = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

num_iterations = 2000
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x, y)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss))

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
