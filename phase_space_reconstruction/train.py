import copy
import os

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from phase_space_reconstruction.losses import MENTLoss, MAELoss
from phase_space_reconstruction.modeling import (
    ImageDataset,
    ImageDataset3D,
    InitialBeam,
    NNTransform,
    PhaseSpaceReconstructionModel,
    PhaseSpaceReconstructionModel3D,
    PhaseSpaceReconstructionModel3D_2screens,
    SextPhaseSpaceReconstructionModel,
)


def train_1d_scan(
    train_dset,
    lattice,
    p0c,
    screen,
    scan_quad_id=0,
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_as=None,
    lambda_=1e11,
    batch_size=10,
):
    """
    Trains beam model by scanning an arbitrary lattice.

    Parameters
    ----------
    train_data: ImageDataset
        training data

    lattice: bmadx TorchLattice
        diagnostics lattice. First element is the scanned quad.

    screen: ImageDiagnostic
        screen diagnostics

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """

    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    ks = train_dset.k.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)

    train_dset_device = ImageDataset(ks, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = PhaseSpaceReconstructionModel(lattice.copy(), screen, nn_beam)

    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(lambda_))

    for i in range(n_epochs):
        for elem in train_dataloader:
            k, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(k, scan_quad_id)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    model = model.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_as is not None:
        torch.save(predicted_beam, save_as)

    return predicted_beam


def train_1d_scan_sext(
    train_dset,
    lattice,
    p0c,
    screen,
    scan_quad_id=0,
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_as=None,
    lambda_=1e11,
    batch_size=10,
):
    """
    Trains beam model by scanning an arbitrary lattice.

    Parameters
    ----------
    train_data: ImageDataset
        training data

    lattice: bmadx TorchLattice
        diagnostics lattice. First element is the scanned quad.

    screen: ImageDiagnostic
        screen diagnostics

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """

    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    ks = train_dset.k.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)

    train_dset_device = ImageDataset(ks, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = SextPhaseSpaceReconstructionModel(lattice.copy(), screen, nn_beam)

    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(lambda_))

    for i in range(n_epochs):
        for elem in train_dataloader:
            k, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(k, scan_quad_id)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    model = model.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_as is not None:
        torch.save(predicted_beam, save_as)

    return predicted_beam


def train_1d_scan_multi_gpu(
    train_dset,
    lattice,
    p0c,
    screen,
    scan_quad_id=0,
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_as=None,
    lambda_=1e11,
    batch_size=10,
):
    """
    Trains beam model by scanning an arbitrary lattice.

    Parameters
    ----------
    train_data: ImageDataset
        training data

    lattice: bmadx TorchLattice
        diagnostics lattice. First element is the scanned quad.

    screen: ImageDiagnostic
        screen diagnostics

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """

    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    ks = train_dset.k.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)

    train_dset_device = ImageDataset(ks, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = PhaseSpaceReconstructionModel(lattice.copy(), screen, nn_beam)

    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(lambda_))
    loss_fn = loss_fn.to(DEVICE)

    for i in range(n_epochs):
        for elem in train_dataloader:
            k, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(k, scan_quad_id)
            loss = loss_fn(output, target_images)
            loss.mean().backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    model = model.module.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_as is not None:
        torch.save(predicted_beam, save_as)

    return predicted_beam


def train_3d_scan(
    train_dset,
    lattice,
    p0c,
    screen,
    ids,
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_dir=None,
    batch_size=10,
    nn_transform=None,
    distribution_dump_frequency=1000,
    distribution_dump_n_particles=100_000,
    use_decay=False
):
    """
    Trains 6D phase space reconstruction model by using 3D scan data.

    Parameters
    ----------
    train_dset: ImageDataset
        training data.

    lattice: bmadx TorchLattice
        6D diagnostics lattice with quadrupole, TDC and dipole

    p0c: float
        beam momentum

    screen: ImageDiagnostic
        Screen (same pixel size and dimensions for both dipole on and off)

    ids: list of ints
        Indices of the elements to be scanned: [quad_id, tdc_id, dipole_id]

    n_epochs: int
        number of epochs for the optimizer

    device: 'cpu' or 'cuda:0'
        device to train the model on

    n_particles: int
        number of particles in the reconstructed beam

    save_as: str or None
        path to save the reconstructed beam

    lambda_: float
        image divergence parameter for the loss function

    batch_size: int
        batch size for the dataloader

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """

    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    params = train_dset.params.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)

    train_dset_device = ImageDataset3D(params, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = nn_transform or NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = PhaseSpaceReconstructionModel3D(lattice.copy(), screen, nn_beam)

    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if use_decay:
        gamma = 0.999  # final learning rate will be gamma * lr
        scheduler = ExponentialLR(optimizer, gamma)
    loss_fn = MAELoss()

    for i in range(n_epochs + 1):
        for elem in train_dataloader:
            params_i, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(params_i, ids)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

        # dump current particle distribution to file
        if i % distribution_dump_frequency == 0:
            if save_dir is not None:
                model_copy = copy.deepcopy(model).to("cpu")
                model_copy.beam.set_base_beam(
                    distribution_dump_n_particles, p0c=torch.tensor(p0c)
                )
                torch.save(
                    model_copy.beam.forward().detach_clone(),
                    os.path.join(save_dir, f"dist_{i}.pt"),
                )
        if use_decay:
            scheduler.step()
            if i % 100 == 0:
                print(scheduler.get_last_lr())

    model = model.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_dir is not None:
        torch.save(predicted_beam, "3d_scan_result.pt")

    return predicted_beam, copy.deepcopy(model)


def train_3d_scan_parallel_gpus(
    train_dset,
    lattice,
    p0c,
    screen,
    ids=[0, 2, 4],
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_as=None,
    lambda_=1e11,
    batch_size=10,
):
    """
    Trains beam model by scanning an arbitrary lattice.
    Note: as of now, the quadrupole that is scanned should
    be the first element of the lattice.

    Parameters
    ----------
    train_data: ImageDataset
        training data

    lattice: bmadx TorchLattice
        diagnostics lattice. First element is the scanned quad.

    screen: ImageDiagnostic
        screen diagnostics

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """

    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    params = train_dset.params.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)
    scan_ids = torch.tensor(ids).to(DEVICE)

    train_dset_device = ImageDataset3D(params, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = PhaseSpaceReconstructionModel3D(lattice.copy(), screen, nn_beam)

    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(lambda_))
    # loss_fn = torch.nn.DataParallel(loss_fn)
    loss_fn = loss_fn.to(DEVICE)

    for i in range(n_epochs):
        for elem in train_dataloader:
            params_i, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(params_i, ids)
            loss = loss_fn(output, target_images)
            loss.mean().backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    model = model.module.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_as is not None:
        torch.save(predicted_beam, save_as)

    return predicted_beam


### TEST ###


def train_3d_scan_2screens(
    train_dset,
    lattice0,
    lattice1,
    p0c,
    screen0,
    screen1,
    ids,
    n_epochs=100,
    device="cpu",
    n_particles=10_000,
    save_dir=None,
    batch_size=10,
    nn_transform=None,
    distribution_dump_frequency=1000,
    distribution_dump_n_particles=100_000,
    use_decay=False
):
    """
    Trains 6D phase space reconstruction model by using 3D scan data.

    Parameters
    ----------
    train_dset: ImageDataset
        training data.
        train_dset.images should be a 6D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of images per parameter configuration,
        screen width in pixels,
        screen height in pixels]
        train_dset.params should be a 4D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of scanning elements (3: quad, tdc, dipole) ]
    lattice: bmadx TorchLattice
        6D diagnostics lattice with quadrupole, TDC and dipole
    p0c: float
        beam momentum
    screen0: ImageDiagnostic
        Screen corresponding to dipole off
    screen1: ImageDiagnostic
        Screen corresponding to dipole on
    ids: list of ints
        Indices of the elements to be scanned: [quad_id, tdc_id, dipole_id]
    n_epochs: int
        number of epochs for the optimizer
    device: 'cpu' or 'cuda:0'
        device to train the model on
    n_particles: int
        number of particles in the reconstructed beam
    save_as: str or None
        path to save the reconstructed beam
    lambda_: float
        image divergence parameter for the loss function
    batch_size: int
        batch size for the dataloader

    Returns
    -------
    predicted_beam: bmadx Beam
        reconstructed beam

    """
    # Device selection:
    DEVICE = torch.device(device)
    print(f"Using device: {DEVICE}")

    params = train_dset.params.to(DEVICE)
    imgs = train_dset.images.to(DEVICE)
    n_imgs_per_param = imgs.shape[-3]

    train_dset_device = ImageDataset3D(params, imgs)
    train_dataloader = DataLoader(
        train_dset_device, batch_size=batch_size, shuffle=True
    )

    # create phase space reconstruction model
    nn_transformer = nn_transform or NNTransform(2, 20, output_scale=1e-2)
    nn_beam = InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(p0c),
    )
    model = PhaseSpaceReconstructionModel3D_2screens(
        lattice0.copy(), lattice1.copy(), screen0, screen1, nn_beam
    )

    model = model.to(DEVICE)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    if use_decay:
        gamma = 0.999  # final learning rate will be gamma * lr
        scheduler = ExponentialLR(optimizer, gamma)
    loss_fn = MAELoss()

    for i in range(n_epochs + 1):
        for elem in train_dataloader:
            params_i, target_images = elem[0], elem[1]
            optimizer.zero_grad()
            output = model(params_i, n_imgs_per_param, ids)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

        # dump current particle distribution to file
        if i % distribution_dump_frequency == 0:
            if save_dir is not None:
                model_copy = copy.deepcopy(model).to("cpu")
                model_copy.beam.set_base_beam(
                    distribution_dump_n_particles, p0c=torch.tensor(p0c)
                )
                torch.save(
                    model_copy.beam.forward().detach_clone(),
                    os.path.join(save_dir, f"dist_{i}.pt"),
                )
        if use_decay:
            scheduler.step()
            if i % 100 == 0:
                print(scheduler.get_last_lr())

    model = model.to("cpu")

    predicted_beam = model.beam.forward().detach_clone()

    if save_dir is not None:
        torch.save(predicted_beam, "3d_scan_result.pt")

    return predicted_beam, copy.deepcopy(model)
