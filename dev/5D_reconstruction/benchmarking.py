import time

import torch
from bmadx.bmad_torch.track_torch import (
    Beam,
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
)
from matplotlib import pyplot as plt

from phase_space_reconstruction import modeling
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import predict_images


def define_beamline(k, k_tdc):
    s1 = TorchQuadrupole(
        torch.tensor(0.1), torch.tensor(10.0), TILT=torch.tensor(3.14 / 4)
    )
    d0 = TorchDrift(torch.tensor(0.5))

    q1 = TorchQuadrupole(torch.tensor(0.1), k)
    d1 = TorchDrift(torch.tensor(0.5))

    tdc1 = TorchCrabCavity(torch.tensor(0.3), k_tdc, torch.tensor(1.3e9))
    d2 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([s1, d0, q1, d1, tdc1, d2])

    return lattice


def create_test_beam(N=10000):
    beam_cov = torch.eye(6) * 1e-6
    particles = torch.distributions.MultivariateNormal(torch.zeros(6), beam_cov).sample(
        [N]
    )
    p0c = torch.tensor(10.0e6)
    beam = Beam(particles, p0c)

    return beam


if __name__ == "__main__":
    # generate synthetic test images
    k = torch.linspace(-10, 20, 3).unsqueeze(-1)
    k_tdc = torch.tensor(1e6)
    lattice = define_beamline(k, k_tdc)
    use_cuda = True

    bins = torch.linspace(-25, 25, 100) * 1e-3
    gt_screen = ImageDiagnostic(bins)

    gt_beam = create_test_beam(10000)
    test_images = predict_images(gt_beam, lattice, gt_screen)

    for ele in test_images:
        fig, ax = plt.subplots()
        ax.pcolor(*gt_screen.mesh, ele.detach())

    # create NN beam
    nn_transformer = modeling.NNTransform(2, 20, output_scale=1e-3)
    nn_beam = modeling.InitialBeam(
        nn_transformer,
        Beam(
            torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)).sample(
                [10]
            ),
            p0c=torch.tensor(10e6),
        ),
    )
    pred_screen = ImageDiagnostic(gt_screen.bins, torch.tensor(0.001))

    for name, val in nn_beam.named_parameters():
        if val.requires_grad:
            print(name)

    # do optimization of NN parameters
    optim = torch.optim.Adam(nn_beam.parameters(), lr=1e-2)
    loss_fn = MENTLoss(torch.tensor(1e14))

    if use_cuda:
        nn_beam = nn_beam.cuda()
        lattice = lattice.cuda()
        pred_screen = pred_screen.cuda()
        test_images = test_images.cuda()
        loss_fn = loss_fn.cuda()

    for i in range(2):
        optim.zero_grad(None)
        start = time.time()
        beam = nn_beam()
        ch1 = time.time()

        # make image predictions
        predicted_images = predict_images(beam, lattice, pred_screen)
        ch2 = time.time()

        # use mse loss function
        loss = loss_fn(predicted_images, test_images, nn_beam.get_entropy(beam)) / 1e5
        ch3 = time.time()
        loss.backward()
        ch4 = time.time()
        if i % 100 == 1:
            print(loss)
            print(ch1 - start)
            print(ch2 - ch1)
            print(ch3 - ch2)
            print(ch4 - ch3)


        optim.step()

    # produce predictions
    predicted_images = predict_images(nn_beam(), lattice, pred_screen)
    for ele in predicted_images:
        fig, ax = plt.subplots()
        ax.pcolor(*pred_screen.mesh.cpu(), ele.cpu().detach())

    plt.show()
