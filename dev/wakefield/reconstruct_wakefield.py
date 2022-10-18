import torch

from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

def wake_function(z, z0, k=2 * torch.pi):
    return -torch.cos(k * (z - z0)) * (torch.tanh((z - z0)  / 0.01) + 1) / 2


class Wakefield(Module):
    def __init__(self):
        super(Wakefield, self).__init__()
        dist = torch.distributions.Normal(0.0, 1.0)
        self.register_parameter("particle_z", Parameter(dist.sample([100, 1])))

    def calculate_wakefield(self, z):
        wake = wake_function(z, self.particle_z)
        total_wake = torch.sum(wake, dim=0)

        return total_wake

def target_wake(z):
    return (torch.tanh(z / 0.01) + 1) / 2 *-250

if __name__ == '__main__':
    model = Wakefield()

    z = torch.linspace(-2.0, 2.0, 200)
    z_test = z[:125]

    # target wakefield
    target = target_wake(z_test)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.00001
    )  # Includes GaussianLikelihood parameters

    if 1:
        for i in range(10000):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model.calculate_wakefield(z_test)
            # Calc loss and backprop gradients
            loss = mse_loss(output, target)
            loss.backward()
            if not i % 1000:
                print(loss)
            optimizer.step()

    # calc entire wakefield
    total_wake = model.calculate_wakefield(z)

    fig, ax = plt.subplots()
    ax.plot(z, total_wake.detach())
    ax.plot(z_test, target)
    axb = ax.twinx()
    axb.hist(model.particle_z.detach().numpy(), alpha=0.25)

    plt.show()