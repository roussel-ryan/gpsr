import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import zuko

from cheetah.particles import ParticleBeam
from cheetah.utils.bmadx import bmad_to_cheetah_coords

from gpsr.modeling import GPSR
from gpsr.modeling import GPSRLattice
from gpsr.modeling import GPSRQuadScanLattice
from gpsr.modeling import BeamGenerator
from gpsr.datasets import QuadScanDataset
from gpsr.datasets import split_dataset
from gpsr.losses import mae_loss
from gpsr.losses import normalize_images


# Command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=10_000)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--prior-scale", type=float, default=1.5)
parser.add_argument("--penalty-min", type=float, default=1000.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--penalty-step", type=float, default=0.0)
parser.add_argument("--penalty-scale", type=float, default=2.0)
parser.add_argument("--eval-nsamp", type=int, default=100_000)
args = parser.parse_args()


# Data
# --------------------------------------------------------------------------------------

filename =  "example_data/example_datasets/reconstruction_4D.dset"
dset = torch.load(filename, weights_only=False)

train_k_ids = np.arange(0, len(dset.parameters), 2)
train_dset, test_dset = split_dataset(dset, train_k_ids)


# Lattice
# --------------------------------------------------------------------------------------

p0c = 43.36e+06  # reference particle momentum [eV/c]
gpsr_lattice = GPSRQuadScanLattice(l_quad=0.1, l_drift=1.0, screen=train_dset.screen)


# Model
# --------------------------------------------------------------------------------------

class FlowBeamGenerator(BeamGenerator):
    def __init__(
        self,
        n_particles: int,
        energy: float,
        flow: zuko.flows.Flow,
        prior: torch.distributions.Distribution = None,
        cov_matrix: torch.Tensor = None,
    ) -> None:
        super(FlowBeamGenerator, self).__init__()

        self.ndim = 6
        self.n_particles = n_particles

        self.flow = flow
        self.prior = prior

        self.unnorm_matrix = torch.eye(ndim)
        if cov_matrix is not None:
            self.unnorm_matrix = torch.linalg.cholesky(cov_matrix)

        self.register_buffer("beam_energy", torch.tensor(energy))
        self.register_buffer("particle_charges", torch.tensor(1.0))
        self.register_buffer("rest_mass", torch.tensor(0.511e6))
    
    def forward(self) -> tuple[ParticleBeam, torch.Tensor]:
        z, log_p = self.flow().rsample_and_log_prob((self.n_particles,))
        log_q = self.prior.log_prob(z)
        entropy = torch.mean(log_p - log_q)

        x = torch.matmul(z, self.unnorm_matrix.T)
        x = bmad_to_cheetah_coords(x, self.beam_energy, self.rest_mass)
        beam = ParticleBeam(*x, particle_charges=self.particle_charges)
        return (beam, entropy)
    

class MyGPSR(GPSR):
    def __init__(self, beam_generator: FlowBeamGenerator, lattice: GPSRLattice) -> None:
        super().__init__(beam_generator=beam_generator, lattice=lattice)

    def forward(self, x: torch.Tensor) -> dict:
        beam, entropy = self.beam_generator()
        self.lattice.set_lattice_parameters(x)
        predictions = self.lattice.track_and_observe(beam)

        results = {}
        results["entropy"] = entropy
        results["predictions"] = predictions

        return results
    

ndim = 6
cov_matrix = torch.eye(ndim) * 0.01 ** 2

flow = zuko.flows.NSF(features=ndim, transforms=3, hidden_features=(3 * [64]))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)

prior_loc = torch.zeros(ndim)
prior_cov = torch.eye(ndim) * args.prior_scale ** 2
prior = torch.distributions.MultivariateNormal(prior_loc, prior_cov)  # in norm space

beam_generator = FlowBeamGenerator(
    flow=flow, 
    prior=prior, 
    cov_matrix=cov_matrix,
    n_particles=args.nsamp, 
    energy=p0c,
)
gpsr_model = MyGPSR(beam_generator=beam_generator, lattice=gpsr_lattice)


# Training
# --------------------------------------------------------------------------------------

def train_epoch(gpsr_model, dset, lr: float, penalty: float, iterations: int) -> None:    
    optimizer = torch.optim.Adam(gpsr_model.beam_generator.flow.parameters(), lr=lr)

    history = {}
    history["loss"] = []
    history["loss_reg"] = []
    history["loss_pred"] = []
    
    for iteration in range(iterations):
        optimizer.zero_grad()

        output = gpsr_model(dset.parameters)

        loss_reg = 0.0
        loss_reg += output["entropy"]

        y_meas = [normalize_images(y) for y in dset.observations]
        y_pred = [normalize_images(y) for y in output["predictions"]]
        diff = [mae_loss(_y_meas, _y_pred) for _y_meas, _y_pred in zip(y_meas, y_pred)]

        loss_pred = 0.0
        if len(diff) > 1:
            loss_pred += torch.add(*diff) / len(diff)
        else:
            loss_pred += diff[0]

        loss = loss_reg + loss_pred * penalty

        loss.backward()
        optimizer.step()

        print(f"iter={iteration} loss_reg={loss_reg.item():0.3e} loss_pred={loss_pred.item():0.3e}")

        history["loss"].append(loss.item())
        history["loss_reg"].append(loss_reg.item())
        history["loss_pred"].append(loss_pred.item())

    return history


penalty = args.penalty_min
for epoch in range(args.epochs):
    print("EPOCH={}".format(epoch))
    print("penalty={}".format(penalty))


    history = train_epoch(
        gpsr_model=gpsr_model,
        dset=train_dset,
        lr=args.lr,
        penalty=penalty,
        iterations=args.iters,
    )
    penalty *= args.penalty_scale
    penalty += args.penalty_step
    if args.penalty_max is not None:
        penalty = min(penalty, args.penalty_max)

    with torch.no_grad():
        # Set number of particles for evaluation
        gpsr_model.beam_generator.n_particles = args.eval_nsamp

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(history["loss_reg"])
        plt.show()

        # Plot data
        results = gpsr_model(train_dset.parameters)
        pred = results["predictions"]
        pred_dset = QuadScanDataset(train_dset.parameters, pred[0].detach(), train_dset.screen)

        fig, ax = train_dset.plot_data(overlay_data=pred_dset)
        fig.set_size_inches(20, 3)
        plt.show()

        # Plot beam
        beam, _ = gpsr_model.beam_generator()
        beam.plot_distribution(
            bins=50,
            plot_2d_kws=dict(
                pcolormesh_kws=dict(cmap="viridis"),
            ),
        )
        plt.show()

        # Set number of particles for training
        gpsr_model.beam_generator.n_particles = args.nsamp
