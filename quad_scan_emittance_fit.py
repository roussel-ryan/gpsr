import torch.nn
from torch.nn.functional import mse_loss


def beam_size_squared(k, d, l, s11, s12, s22):
    return (
        (1.0 + k * d * l) ** 2 * s11 + 2.0 * (1.0 + d * l * k) * d * s12 + d ** 2 * s22
    )


class EmittanceQuadScan(torch.nn.Module):
    def __init__(self, d, l):
        super(EmittanceQuadScan, self).__init__()

        self.register_buffer("d", d)
        self.register_buffer("l", l)

        self.register_parameter("s11", torch.nn.Parameter(torch.ones(1) * 1e-7))
        self.register_parameter("s12", torch.nn.Parameter(torch.ones(1) * 1e-7))
        self.register_parameter("s22", torch.nn.Parameter(torch.ones(1) * 1e-7))

    def forward(self, k):
        return beam_size_squared(k, self.d, self.l, self.s11, self.s12, self.s22)

    @property
    def emittance(self):
        return torch.sqrt(self.s11 * self.s22 - self.s12 ** 2)


class ReparameterizedEmittanceQuadScan(torch.nn.Module):
    def __init__(self, d, l):
        super(ReparameterizedEmittanceQuadScan, self).__init__()

        self.register_buffer("d", d)
        self.register_buffer("l", l)

        # note: we constrain c to be in the domain [-1,1] using an atan transform

        self.register_parameter("raw_c", torch.nn.Parameter(torch.zeros(1).to(d)))
        self.register_parameter(
            "lambda_", torch.nn.Parameter(torch.ones(2).to(d) * 1e-3)
        )

    def forward(self, k):
        bm = self.beam_matrix
        return beam_size_squared(k, self.d, self.l, bm[0, 0], bm[1, 0], bm[1, 1])

    @property
    def beam_matrix(self):
        c = self.c
        c0 = torch.tensor([1.0, 0.0]).to(c).unsqueeze(0)
        c1 = torch.cat((c, torch.sqrt(1 - c ** 2))).unsqueeze(0)
        C = torch.cat((c0, c1))

        L = torch.diag(self.lambda_) @ C
        return L @ L.T

    @beam_matrix.setter
    def beam_matrix(self, value):
        # check to make sure value is a valid beam matrix
        torch.linalg.cholesky(value)

        s11 = value[0, 0]
        s12 = value[1, 0]
        s22 = value[1, 1]

        new_lambda = torch.tensor([torch.sqrt(s11), torch.sqrt(s22)])
        self.lambda_.data = new_lambda

        mult = -1 if s12 < 0 else 1
        self.c = (mult * s12 / (self.lambda_[0] * self.lambda_[1])).unsqueeze(0)

    @property
    def c(self):
        return torch.atan(self.raw_c)

    @c.setter
    def c(self, value):
        self.raw_c.data = torch.tan(value)

    @property
    def emittance(self):
        return torch.det(self.beam_matrix)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    distance = torch.tensor(1.0).double()
    q_len = torch.tensor(0.1).double()
    s11 = torch.tensor(3e-6).double()
    s12 = torch.tensor(1.5e-6).double()
    s22 = torch.tensor(2e-6).double()
    emit = torch.sqrt(s11 * s22 - s12 ** 2)
    print(emit)

    k = torch.linspace(-50, 10, 10)
    gty = beam_size_squared(k, distance, q_len, s11, s12, s22)

    # fig,ax =plt.subplots()
    # ax.plot(k, gty)

    model = ReparameterizedEmittanceQuadScan(distance, q_len)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )  # Includes GaussianLikelihood parameters

    if 1:
        for i in range(10000):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(k)
            # Calc loss and backprop gradients
            loss = mse_loss(output*1e6, gty*1e6)
            loss.backward()
            if not i % 1000:
                print(loss)
            optimizer.step()

        print(list(model.named_parameters()))
        print(model.beam_matrix)

    with torch.no_grad():
        #model.beam_matrix = torch.tensor(((s11, s12), (s12, s22)))
        #print(list(model.named_parameters()))

        pred_y = model(k)

        plt.plot(k, gty, "o")
        plt.plot(k, pred_y.detach())

    plt.show()
