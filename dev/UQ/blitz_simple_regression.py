import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.nn import Linear, Module
from tqdm import trange


class Regressor(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(1, 100)
        self.linear2 = Linear(100, 1)

    def forward(self, X):
        x = self.linear1(X)
        x = F.relu(x)
        return self.linear2(x)


# create model
@variational_estimator
class BayesianRegressor(Module):
    def __init__(self):
        super().__init__()
        self.blinear1 = BayesianLinear(
            1, 100
        )
        self.blinear2 = BayesianLinear(
            100, 1
        )

    def forward(self, X):
        x = self.blinear1(X)
        x = F.relu(x)
        return self.blinear2(x)


def f(x):
    return x + 0.3 * torch.sin(2 * 3.14 * x) + 0.3 * torch.sin(4 * 3.14 * x)


# generate training data
train_X = torch.rand(100, 1)
train_Y = f(train_X) + 0.2 * torch.randn_like(train_X)

# test results
test_X = torch.linspace(-0.5, 1.5, 200).unsqueeze(1)
gt_Y = f(test_X)

# train model
use_bayes = True
if use_bayes:
    model = BayesianRegressor()
else:
    model = Regressor()

optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for i in trange(2000):
    optim.zero_grad()

    if use_bayes:
        loss = model.sample_elbo(
            inputs=train_X,
            labels=train_Y,
            criterion=loss_fn,
            sample_nbr=3,
            complexity_cost_weight=1e-4 / len(train_X),
        )
    else:
        loss = loss_fn(model(train_X), train_Y)

    loss.backward()
    optim.step()




# evaluate model
with torch.no_grad():
    preds = torch.stack([model(test_X) for i in range(50)])
    mean = torch.mean(preds, dim=0).flatten()
    std = torch.std(preds, dim=0).flatten()
    l = mean - std*3
    u = mean + std*3


fig, ax = plt.subplots()
ax.plot(test_X, gt_Y)
ax.plot(train_X, train_Y, "+")

ax.plot(test_X.flatten(), mean)
ax.fill_between(test_X.flatten(), l, u, alpha=0.25)

plt.show()
