import os

import torch
from torch.utils.data import DataLoader

from phase_space_reconstruction.beamlines import quad_scan_lattice
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import (
    NNTransform,
    InitialBeam,
    PhaseSpaceReconstructionModel,
    ImageDataset
    )


def train_model(
        scan_data, 
        n_epochs = 100,
        device = 'cpu',
        save_as = None
        ):
    
    # Device selection: 
    DEVICE = torch.device(device)
    print(f'Using device: {DEVICE}')

    all_k = scan_data['quad_strengths'].to(DEVICE)
    all_images = scan_data['images'].to(DEVICE)
    bins = scan_data['bins'].to(DEVICE)
    
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
    lattice = quad_scan_lattice()

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

    prediction = {
        'beam': model.beam.forward(),
        'images': model.forward(all_k)[0],
        'train_ids': train_ids,
        'test_ids': test_ids
    }

    if save_as is not None:
        torch.save(prediction, save_as)
    
    return prediction 
    
