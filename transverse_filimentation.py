import time

import matplotlib.pyplot as plt
from distgen import Generator

# create beam
from distgen.physical_constants import unit_registry as unit

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
    "cosy": ycos,
    "order": ["twiss", "cosy"],
}
gen.run()
particles_3 = gen.particles

particles = particles_1 + particles_2 + particles_3

particles.plot("x", "px")
particles.plot("x", "y")

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
    torch.tensor(particles.xp, **tkwargs),
    torch.tensor(particles.y, **tkwargs),
    torch.tensor(particles.yp, **tkwargs),
    torch.tensor(particles.z, **tkwargs),
    (torch.tensor(particles.pz, **tkwargs) - defaults["p0c"]) / defaults["p0c"],
    **defaults
)

from normalizing_flow import get_images, image_difference_loss

# get ground truth images
sizes = []
k_in = torch.linspace(-10, 20, 10, **tkwargs).unsqueeze(1)
bins = torch.linspace(-50, 50, 100, **tkwargs) * 1e-3
bandwidth = torch.tensor(5e-4, **tkwargs)

# define batched lattice
quad = Quadrupole(torch.tensor(0.1, **tkwargs), K1=k_in)
drift = Drift(L=torch.tensor(1.0, **tkwargs))
lattice = Lattice([quad, drift], torch)

ground_truth_output_beams = lattice(ground_truth_in)[-1]
ground_truth_images = get_images(ground_truth_output_beams, bins, bandwidth)

# for image in ground_truth_images[:5]:
#    fig, ax = plt.subplots()
#    xx = torch.meshgrid(bins, bins)
#    ax.pcolor(xx[0].cpu(), xx[1].cpu(), image.cpu().detach())


# define a unit multivariate normal
from torch.distributions import MultivariateNormal

normal_dist = MultivariateNormal(torch.zeros(6), torch.eye(6))
normal_samples = normal_dist.sample([10000]).to(**tkwargs)

# define nonparametric transform
from normalizing_flow import NonparametricTransform, track_in_quad

tnf = NonparametricTransform()
tnf.to(tkwargs["device"])

# plot initial beam after transform
p_in_guess = Particle(*tnf(normal_samples).T, **defaults)
# plt.hist2d(
#    p_in_guess.x.cpu().detach().numpy(), p_in_guess.px.cpu().detach().numpy(), bins=50
# )

# plt.figure()
# p_out_guess = lattice(p_in_guess)[-1]
# plt.hist2d(
#    p_out_guess.x[0].cpu().detach().numpy(),
#    p_out_guess.px[0].cpu().detach().numpy(),
#    bins=50)

# preform optimization with Adam to generate a beam with zero centroid
from normalizing_flow import (
    beam_position_loss,
    image_difference_loss,
    zero_centroid_loss,
)

optim = torch.optim.Adam(tnf.parameters(), lr=0.01)
n_iter = 5000
losses = []
image_difference_losses = []
beam_position_losses = []

start = time.time()
for i in range(n_iter):
    optim.zero_grad(set_to_none=True)
    guess_dist = Particle(*tnf(normal_samples).T, **defaults)

    loss = image_difference_loss(
        guess_dist, ground_truth_images, lattice, bins=bins, bandwidth=bandwidth
    )
    losses += [loss.cpu().detach()]
    loss.backward()

    optim.step()
finish = time.time()

print(f"{(finish - start) / n_iter} s per step")

fig, ax = plt.subplots()
ax.semilogy(losses)

guess_dist = Particle(*tnf(normal_samples).T, **defaults)
image_difference_loss(
    guess_dist,
    ground_truth_images,
    lattice,
    bins=bins,
    bandwidth=bandwidth,
    plot_images=True,
)

# plot initial beam after transform
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
ax[0].hist2d(
    guess_dist.x.cpu().detach().numpy(), guess_dist.px.detach().cpu().numpy(), bins=100
)
ax[1].hist2d(
    ground_truth_in.x.cpu().detach().numpy(),
    ground_truth_in.px.detach().cpu().numpy(),
    bins=100,
)

# plot initial beam after transform
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
ax[0].hist2d(
    guess_dist.y.cpu().detach().numpy(),
    guess_dist.py.detach().cpu().numpy(), bins=100
)
ax[1].hist2d(
    ground_truth_in.y.cpu().detach().numpy(),
    ground_truth_in.py.detach().cpu().numpy(),
    bins=100,
)

plt.show()
