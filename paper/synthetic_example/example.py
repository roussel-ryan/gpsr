import torch
from torch.autograd import grad
from torch.nn import Parameter


# define a function to optimize
def f(x):
    return x ** 2


def g(x):
    return f(x) - 1


class ConstrainedModel(torch.nn.Module):
    def __init__(self, objective, costraint, x0, l0):
        super(ConstrainedModel, self).__init__()
        self.objective = objective
        self.constraint = costraint

        self.register_parameter("lambda_", Parameter(l0))
        self.register_parameter("x", Parameter(x0))

    def forward(self):
        return self.objective(self.x) + self.lambda_ * self.constraint(self.x)


def optimize(model, lr=0.01, n_steps=100):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    track = []
    for ii in range(n_steps):
        loss = model()
        z = grad(loss, model.parameters(), create_graph=True)
        grad_loss = torch.norm(torch.cat([ele.unsqueeze(0) for ele in z]))
        grad_loss.backward()

        optim.step()

        optim.zero_grad()
        for ele in z:
            ele.grad = None

        params = list(model.parameters())
        track.append([ele.data.clone() for ele in params] + [grad_loss])

    return track


x0 = torch.tensor(0.01)
lambda_0 = torch.tensor(-9.0)
constrained_model = ConstrainedModel(f, g, x0, lambda_0)

results = optimize(constrained_model, lr=1.1, n_steps=1000)
print(list(constrained_model.parameters()))
