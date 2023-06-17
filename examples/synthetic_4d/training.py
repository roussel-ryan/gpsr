import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from phase_space_reconstruction.histogram import histogram2d

from phase_space_reconstruction import modeling
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import ImageDataset

from beamline import create_4d_diagnostic_beamline
from bmadx.plot import plot_projections

def generate_training_images(true_beam):
    tkwargs = {"device": "cuda", "dtype": torch.float32}

    n_images = 20
    k_in = torch.linspace(-25, 15, n_images, **tkwargs).unsqueeze(1)
    bins = torch.linspace(-30, 30, 200, **tkwargs) * 1e-3

    # do tracking
    train_lattice = create_4d_diagnostic_beamline()
    train_lattice.elements[0].K1.data = k_in

    train_lattice = train_lattice.cuda()
    input_beam = true_beam.cuda()

    output_beam = train_lattice(input_beam)
    screen_data = torch.cat(
        [ele.unsqueeze(-1) for ele in [output_beam.x, output_beam.y]], dim=-1
    ).cpu().float()

    # do histogramming
    images = []
    bins = bins.cpu()
    bin_width = bins[1]-bins[0]
    bandwidth = bin_width.cpu() / 2
    for i in range(n_images):
        hist = histogram2d(screen_data[i].T[0], screen_data[i].T[1], bins, bandwidth)
        images.append(hist)

    images = torch.cat([ele.unsqueeze(0) for ele in images], dim=0)

    for i in range(0, len(images)):
        plt.figure()
        plt.imshow(images[i])
        print(k_in[i], torch.std(output_beam.x[i])**2 * 1e6)

    xx = torch.meshgrid(bins, bins)
    # save image data
    torch.save(images.unsqueeze(1), "train_images.pt")
    torch.save(k_in, "kappa.pt")
    torch.save(bins, "bins.pt")
    torch.save(xx, "xx.pt")


def load_data():
    folder = ""

    all_k = torch.load(folder + "kappa.pt").float().unsqueeze(-1)
    all_images = torch.load(folder + "train_images.pt").float()
    xx = torch.stack(torch.load(folder + "xx.pt"))

    bins = xx[0].T[0]
    gt_beam = torch.load(folder + "ground_truth_dist.pt")

    return all_k, all_images, bins, xx, gt_beam


def train_model():
    # import and organize data for training
    all_k, all_images, bins, xx, gt_beam = load_data()

    if torch.cuda.is_available():
        all_k = all_k.cuda()
        all_images = all_images.cuda()
        bins = bins.cuda()
        gt_beam = gt_beam.cuda()


    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])

    train_dataloader = DataLoader(train_dset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2

    # create phase space reconstruction model
    diagnostic = ImageDiagnostic(bins, bandwidth=bandwidth)
    lattice = create_4d_diagnostic_beamline()

    n_particles = 10000
    nn_transformer = modeling.NNTransform(2, 20, output_scale=1e-2)
    nn_beam = modeling.InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(10.0e6),
    )

    model = modeling.PhaseSpaceReconstructionModel(
        lattice,
        diagnostic,
        nn_beam
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # train model
    n_epochs = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(1e11))

    for i in range(n_epochs):
        for batch_idx, elem in enumerate(train_dataloader):
            k, target_images = elem[0], elem[1]

            optimizer.zero_grad()
            output = model(k)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    # visualize predictions
    image_pred, entropy_pred, cov_pred = model(all_k)
    n_im = len(all_k)
    fig,ax = plt.subplots(2, n_im, sharex="all", sharey="all", figsize=(40,10))
    for i in range(n_im):
        ax[0, i].pcolor(*xx.cpu(), all_images[i].squeeze().detach().cpu())
        ax[1, i].pcolor(*xx.cpu(), image_pred[i].squeeze().detach().cpu())
    ax[0, 2].set_title("ground truth")
    ax[1, 2].set_title("prediction")

    beam_prediction = model.beam.forward()
    plot_projections(beam_prediction.data.cpu().detach().numpy(),
                     labels=['x', 'px', 'y', 'py', 'z', 'pz'],
                     bins=50,
                     background=False)


if __name__ == "__main__":
    train_single_model()
    plt.show()

