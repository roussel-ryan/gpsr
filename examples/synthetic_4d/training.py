import os

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from phase_space_reconstruction.histogram import histogram2d
from phase_space_reconstruction.modeling import NNTransform, InitialBeam, PhaseSpaceReconstructionModel, ImageDataset
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss

from beamline import create_4d_diagnostic_beamline

from bmadx.plot import plot_projections


def generate_images(gt_beam, folder='data', verbose=False):
    tkwargs = {"dtype": torch.float32}

    # scan parameters
    n_images = 20
    quad_ks = torch.linspace(-25, 15, n_images, **tkwargs).unsqueeze(1)
    bins = torch.linspace(-30, 30, 200, **tkwargs) * 1e-3

    # tracking though diagnostics lattice
    lat = create_4d_diagnostic_beamline()
    lat.elements[0].K1.data = quad_ks
    input_beam = gt_beam
    output_beam = lat(input_beam)

    # differentiable histogramming at screen
    images = []
    bins = bins.cpu()
    bin_width = bins[1]-bins[0]
    bandwidth = bin_width.cpu() / 2

    for i in range(n_images):
        hist = histogram2d(output_beam.x[i].detach().cpu(),
                           output_beam.y[i].detach().cpu(),
                           bins, bandwidth)
        images.append(hist)

    images = torch.cat([ele.unsqueeze(0) for ele in images], dim=0)

    # save scan data
    scan_data = {
        'quad_strengths': quad_ks,
        'images': images,
        'bins': bins
    }
   
    torch.save(scan_data, os.path.join(folder,"scan_data.pt"))

    # plot images (if you want)
    if verbose:
        print(f'number of images = {n_images}\n')
        xx = torch.meshgrid(bins, bins, indexing='ij')
        for i in range(0, len(images)):
            print(f'image {i}')
            print(f'k = {quad_ks[i]} 1/m')
            print(f'stdx = {torch.std(output_beam.x[i])**2 * 1e6} mm')
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pcolor(xx[0]*1e3, xx[1]*1e3, images[i])
            ax.set_xlabel('$x$ (mm)')
            ax.set_ylabel('$y$ (mm)')
            plt.show()

    return scan_data


def train_model(n_epochs = 100, folder='data'):
    # Device selection: 
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print(f'Using device: {DEVICE}')

    # import scan data and ground truth beam
    scan_data = torch.load(os.path.join(folder, 'scan_data.pt'))
    gt_beam = torch.load(os.path.join(folder, 'gt_dist.pt'))

    all_k = scan_data['quad_strengths'].to(DEVICE)
    all_images = scan_data['images'].to(DEVICE)
    bins = scan_data['bins'].to(DEVICE)
    
    gt_beam = gt_beam.to(DEVICE)
    
    # divide scan data in training and testing data sets
    n_scan = len(scan_data['quad_strengths'])
    train_ids = [*range(n_scan)[::2]]
    test_ids = [*range(n_scan)[1::2]]
    train_dset = ImageDataset(all_k[train_ids], all_images[train_ids])
    test_dset = ImageDataset(all_k[test_ids], all_images[test_ids])

    train_dataloader = DataLoader(train_dset, batch_size=10, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2

    # create phase space reconstruction model
    diagnostic = ImageDiagnostic(bins, bandwidth=bandwidth)
    lattice = create_4d_diagnostic_beamline()

    n_particles = 10000
    nn_transformer = NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(10.0e6),
    )

    model = PhaseSpaceReconstructionModel(
        lattice,
        diagnostic,
        nn_beam
    )

    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(1e11))

    for i in range(n_epochs):
        for elem in train_dataloader:
            k, target_images = elem[0], elem[1]

            optimizer.zero_grad()
            output = model(k)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    pred_image_data = {
        'quad_strengths': all_k.detach().cpu().numpy(),
        'images': model(all_k)[0].detach().cpu().numpy(),
        'bins': bins.detach().cpu().numpy(),
        'train_ids': train_ids,
        'test_ids': test_ids
    }
    reconstructed_beam = model.beam.forward()

    return reconstructed_beam, pred_image_data
    
