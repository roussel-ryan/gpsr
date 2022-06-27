import time

import matplotlib.pyplot as plt
from distgen import Generator

# create beam
from distgen.physical_constants import unit_registry as unit

from modeling import ImagingModel, NonparametricTransform, train
from visualization import compare_images, plot_reconstructed_phase_space

gen = Generator("beams/gaussian.yaml")
gen.run()
particles_1 = gen.particles

# add a transformation to gen
setstdx = {"type": "set_std x", "sigma_x": {"value": 2, "units": "mm"}}
setstdpx = {"type": "set_std px", "sigma_px": {"value": 0.01, "units": "MeV/c"}}
k = 2*3.14 / (50*unit("mm"))
ycos = {
    "type": "cosine x:y",
    "amplitude": {"value": 30, "units": "mm"},
    "omega": {"value": k.magnitude, "units": str(k.units)},
    "phase": {"value": 90, "units": "deg"}
}

pycos = {
    "type": "cosine y:py",
    "amplitude": {"value": 0.005, "units": "MeV/c"},
    "omega": {"value": k.magnitude, "units": str(k.units)},
    "phase": {"value": 90, "units": "deg"}
}

twiss_x = {
    'type': 'set_twiss x',
    'beta': {'value': 10, 'units': 'm', },
    'alpha': {'value': 5, 'units': ''},
    'emittance': {'value': 20.0, 'units': 'um'}
}

# rot2dxxp = {'type':'rotate2d x:xp', 'angle':{'value':45, 'units':'deg'}}

gen.input["transforms"] = {
    "t1": setstdx,
    "t2": setstdpx,
    "order": ["t1", "t2"],
}
gen.run()
particles_2 = gen.particles

gen.input["transforms"] = {
    "twiss": twiss_x,
    "ycos": ycos,
    "pycos": pycos,
    "order": ["twiss", "ycos", "pycos"],
}
gen.run()
particles_3 = gen.particles

particles = particles_1 + particles_2 + particles_3

particles.plot("x", "y")
particles.plot("x", "px")
particles.plot("y", "py")
particles.plot("z", "pz")


import torch

# transform particles from distgen to BMAD
from track import Drift, Lattice, Particle, Quadrupole

tkwargs = {"device": "cuda", "dtype": torch.float32}

keys = ["x", "px", "y", "py", "z", "pz"]
defaults = {
    "s": torch.tensor(0.0, **tkwargs),
    "p0c": torch.mean(torch.tensor(particles.pz, **tkwargs)),
    "mc2": torch.tensor(particles.mass, **tkwargs),
}

ground_truth_in = Particle(
    torch.tensor(particles.x, **tkwargs),
    torch.tensor(particles.px, **tkwargs) / defaults["p0c"],
    torch.tensor(particles.y, **tkwargs),
    torch.tensor(particles.py, **tkwargs) / defaults["p0c"],
    torch.tensor(particles.z, **tkwargs),
    (torch.tensor(particles.pz, **tkwargs) - defaults["p0c"]) / defaults["p0c"],
    **defaults
)

from normalizing_flow import get_images, image_difference_loss

# get ground truth images
sizes = []
k_in = torch.linspace(-10, 20, 10, **tkwargs).unsqueeze(1)
bins = torch.linspace(-50, 50, 100, **tkwargs) * 1e-3
bandwidth = torch.tensor(1e-3, **tkwargs)

# define batched lattice
quad = Quadrupole(torch.tensor(0.1, **tkwargs), K1=k_in)
drift = Drift(L=torch.tensor(1.0, **tkwargs))
train_lattice = Lattice([quad, drift], torch)

ground_truth_output_beams = train_lattice(ground_truth_in)[-1]
train_images = get_images(ground_truth_output_beams, bins, bandwidth)

transform = NonparametricTransform()
transform.to(tkwargs["device"])

n_particles = 10000
model = ImagingModel(
                transform,
                bins,
                bandwidth,
                torch.tensor(n_particles),
                defaults
            )
model.cuda()

losses = train(model, train_lattice, train_images, 5000, lr=0.01)

fig, ax = plt.subplots()
ax.semilogy(losses)

compare_images(model, train_lattice, train_images)

plot_reconstructed_phase_space("x", "y", [model])
plot_reconstructed_phase_space("x", "px", [model])
plot_reconstructed_phase_space("y", "py", [model])
plot_reconstructed_phase_space("z", "pz", [model])

plt.show()
