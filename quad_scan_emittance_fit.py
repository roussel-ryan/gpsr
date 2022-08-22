import torch.nn
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss


def beam_size_squared(k, d, l, s11, s12, s22):
    return (1.0 + k * d * l) ** 2 * s11 + \
           2.0 * (1.0 + d * l * k) * d * s12 + \
           d ** 2 * s22


class EmittanceQuadScan(torch.nn.Module):
    def __init__(self, k, d, l):
        super(EmittanceQuadScan, self).__init__()
        self.register_buffer("k", k)
        self.register_buffer("d", d)
        self.register_buffer("l", l)

        self.register_parameter("s11", torch.nn.Parameter(torch.ones(1) * 1e-6))
        self.register_parameter("s12", torch.nn.Parameter(torch.ones(1) * 1e-6))
        self.register_parameter("s22", torch.nn.Parameter(torch.ones(1) * 1e-6))

    def forward(self):
        return beam_size_squared(self.k, self.d, self.l, self.s11, self.s12, self.s22)

    @property
    def emittance(self):
        return torch.sqrt(self.s11 * self.s22 - self.s12**2)


if __name__ == '__main__':
    distance = torch.tensor(1.0)
    q_len = torch.tensor(0.1)
    s11 = torch.tensor(3e-6)
    s12 = torch.tensor(1e-6)
    s22 = torch.tensor(2e-6)
    emit = torch.sqrt(s11 * s22 - s12 ** 2)
    print(emit)

    k = torch.linspace(-50, 10, 10)
    gty = beam_size_squared(k, distance, q_len, s11, s12, s22)

    # fig,ax =plt.subplots()
    # ax.plot(k, gty)

    model = EmittanceQuadScan(k, distance, q_len)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1)  # Includes GaussianLikelihood parameters

    for i in range(1000):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model()
        # Calc loss and backprop gradients
        loss = mse_loss(output, gty)
        loss.backward()
        print(loss)
        optimizer.step()

    print(list(model.named_parameters()))

    plt.show()
