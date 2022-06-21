import torch
import time
# transform particles from distgen to BMAD
from track import Particle, Lattice, Drift, Quadrupole, make_track_a_drift, \
    make_track_a_quadrupole


# define a unit multivariate normal
from torch.distributions import MultivariateNormal


def run_quad_track(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([n]).to(**tkwargs)

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }
    p_in = Particle(*normal_samples.T, **defaults)
    quad = Quadrupole(
        torch.tensor(1.0, **tkwargs),
        torch.tensor(1.0, **tkwargs),
    )
    quad_track = make_track_a_quadrupole(torch)

    start = time.time()
    quad_track(p_in, quad)
    finish = time.time()
    print(f"{finish-start}")


def run_drift_track(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([n]).to(**tkwargs) * 1e-3

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }
    p_in = Particle(*normal_samples.T, **defaults)
    drift = Drift(torch.tensor(1.0, **tkwargs))
    drift_track = make_track_a_drift(torch)

    start = time.time()
    drift_track(p_in, drift)
    finish = time.time()
    print(f"{finish-start}")


def run_simple_lattice_track(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([n]).to(**tkwargs) * 1e-3

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }
    p_in = Particle(*normal_samples.T, **defaults)
    drift = Drift(torch.tensor(1.0, **tkwargs))
    drift_track = make_track_a_drift(torch)

    quad = Quadrupole(
        torch.tensor(1.0, **tkwargs),
        torch.tensor(1.0, **tkwargs),
        NUM_STEPS=10
    )
    quad_track = make_track_a_quadrupole(torch)

    tracking_functions = [quad_track, drift_track]
    elements = [quad, drift]

    p_out = p_in
    start = time.time()
    for func, ele in zip(tracking_functions, elements):
        p_out = func(p_out, ele)

    finish = time.time()
    print(f"{finish-start}")


def run_lattice_track(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([n]).to(**tkwargs) * 1e-3

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }
    p_in = Particle(*normal_samples.T, **defaults)
    drift = Drift(torch.tensor(1.0, **tkwargs))

    quad = Quadrupole(
        torch.tensor(1.0, **tkwargs),
        torch.tensor(1.0, **tkwargs),
        NUM_STEPS=10
    )
    lattice = Lattice([quad, drift], torch)

    start = time.time()
    lattice(p_in)
    finish = time.time()
    print(f"{finish-start}")


def run_lattice_track_batched_beam(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([10, n]).to(**tkwargs) * 1e-3

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }
    p_in = Particle(*normal_samples.T, **defaults)
    drift = Drift(torch.tensor(1.0, **tkwargs))

    quad = Quadrupole(
        torch.tensor(1.0, **tkwargs),
        torch.tensor(1.0, **tkwargs),
        NUM_STEPS=10
    )
    lattice = Lattice([quad, drift], torch)

    start = time.time()
    lattice(p_in)
    finish = time.time()
    print(f"{finish-start}")


def run_lattice_track_batched_params(tkwargs, n):
    normal_dist = MultivariateNormal(
        torch.zeros(6), torch.eye(6)
    )
    normal_samples = normal_dist.sample([n, 1]).to(**tkwargs) * 1e-3

    defaults = {
        "s": torch.tensor(0.0, **tkwargs),
        "p0c": torch.tensor(1.0, **tkwargs),
        "mc2": torch.tensor(1.0, **tkwargs),
    }

    p_in = Particle(*normal_samples.reshape(6, -1, 1), **defaults)
    drift = Drift(torch.tensor(1.0, **tkwargs))

    quad = Quadrupole(
        torch.tensor(1.0, **tkwargs),
        torch.linspace(0.0, 1.0, 10, **tkwargs),
        NUM_STEPS=10
    )
    lattice = Lattice([quad, drift], torch)

    start = time.time()
    out = lattice(p_in)
    finish = time.time()
    print(f"{finish-start}")
    print(out[-1].x.shape)


N = 1000000
print("drift")
run_drift_track({"device": "cpu", "dtype": torch.float32}, N)
run_drift_track({"device": "cuda", "dtype": torch.float32}, N)

print("quad")
run_quad_track({"device": "cpu", "dtype": torch.float32}, N)
run_quad_track({"device": "cuda", "dtype": torch.float32}, N)

print("lattice")
run_lattice_track({"device": "cpu", "dtype": torch.float32}, N)
run_lattice_track({"device": "cuda", "dtype": torch.float32}, N)

print("lattice batched")
run_lattice_track_batched_params({"device": "cpu", "dtype": torch.float32}, N)
run_lattice_track_batched_params({"device": "cuda", "dtype": torch.float32}, N)
