import logging
import os

import sys
import time

import torch

sys.path.append("../")
from bmadx.bmad_torch.track_torch import Beam, TorchDrift, TorchLattice, TorchQuadrupole

from phase_space_reconstruction import modeling
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import ImageDataset
from torch.utils.data import DataLoader
from torchensemble import SnapshotEnsembleRegressor

logging.basicConfig(level=logging.INFO)


def create_quad_scan_beamline():
    q1 = TorchQuadrupole(torch.tensor(0.1), torch.tensor(0.0), 5)
    d1 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([q1, d1])
    return lattice


def create_ensemble(bins, bandwidth):
    lattice = create_quad_scan_beamline()
    diagnostic = ImageDiagnostic(bins, bandwidth=bandwidth)
    # create NN beam
    n_particles = 10000
    nn_transformer = modeling.NNTransform(2, 20, output_scale=1e-2)
    nn_beam = modeling.InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(10.0e6),
    )
    module_kwags = {"lattice": lattice, "diagnostic": diagnostic, "beam": nn_beam}

    ensemble = SnapshotEnsembleRegressor(
        estimator=modeling.PhaseSpaceReconstructionModel,
        estimator_args=module_kwags,
        n_estimators=20,
    )

    return ensemble


def load_data():
    folder = "synthetic_beam/"

    all_k = torch.load(folder + "kappa.pt").float().unsqueeze(-1)
    all_images = torch.load(folder + "train_images.pt").float()
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0]
    gt_beam = torch.load(folder + "ground_truth_dist.pt")

    return all_k, all_images, bins, xx, gt_beam


def create_datasets(all_k, all_images, save_dir):
    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])
    torch.save(train_dset, save_dir + "/train.dset")
    torch.save(test_dset, save_dir + "/test.dset")

    return train_dset, test_dset


if __name__ == "__main__":
    save_dir = "double_small_emittance_case_cov_term_no_energy_spread"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    tkwargs = {"dtype": torch.float}
    all_k, all_images, bins, xx, gt_beam = load_data()
    torch.save(gt_beam, save_dir + "/gt_beam.pt")
    train_dset, test_dset = create_datasets(all_k, all_images, save_dir)
    print(len(train_dset))

    train_dataloader = DataLoader(train_dset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2
    ensemble = create_ensemble(bins, bandwidth)

    criterion = MENTLoss(
        torch.tensor(1e11), gamma_=torch.tensor(0.001), alpha_=torch.tensor(0.1)
    )
    # criterion = MENTLoss(torch.tensor(1e3), gamma_=torch.tensor(0.01))
    ensemble.set_criterion(criterion)

    n_epochs = 10000
    ensemble.set_optimizer("Adam", lr=1e-2)
    # with torch.autograd.detect_anomaly():
    start = time.time()
    ensemble.fit(
        train_dataloader, epochs=n_epochs, save_dir=save_dir, lr_clip=[1e-3, 10]
    )
    print(f"runtime: {time.time() - start}")
    torch.save(criterion.loss_record, save_dir + "/loss_log.pt")
