import matplotlib.pyplot as plt
from distgen import Generator

# create beam
gen = Generator("beams/gaussian.yaml")
gen.run()
particles_1 = gen.particles

# add a transformation to gen
setstdx = {"type": "set_std x", "sigma_x": {"value": 3, "units": "mm"}}
setstdpx = {"type": "set_std px", "sigma_px": {"value": 0.01, "units": "MeV/c"}}
transpx = {"type": "translate x", "delta": {"value": 0.01, "units": "m"}}

# rot2dxxp = {'type':'rotate2d x:xp', 'angle':{'value':45, 'units':'deg'}}

gen.input["transforms"] = {
    "t1": setstdx,
    "t2": setstdpx,
    "t3": transpx,
    "order": ["t1", "t2", "t3"],
}
gen.run()
particles_2 = gen.particles

particles = particles_1 + particles_2

import torch

# transform particles from distgen to BMAD
from track import Particle

tkwargs = {"device": "cuda", "dtype": torch.float32}

keys = ["x", "px", "y", "py", "z", "pz"]
defaults = {
    "s": torch.tensor(0.0, **tkwargs),
    "p0c": torch.mean(torch.tensor(particles.pz, **tkwargs)),
    "mc2": torch.tensor(particles.mass, **tkwargs),
}
print(defaults["p0c"])

ground_truth_in = Particle(
    torch.tensor(particles.x, **tkwargs),
    torch.tensor(particles.xp, **tkwargs),
    torch.tensor(particles.y, **tkwargs),
    torch.tensor(particles.yp, **tkwargs),
    torch.tensor(particles.z, **tkwargs),
    (torch.tensor(particles.pz, **tkwargs) - defaults["p0c"]) / defaults["p0c"],
    **defaults
)

from normalizing_flow import get_images, get_output_beams, image_difference_loss

# get ground truth images
sizes = []
k_in = torch.linspace(-10, 20, 10, **tkwargs)
bins = torch.linspace(-50, 50, 100, **tkwargs) * 1e-3
bandwidth = torch.tensor(1e-3, **tkwargs)
ground_truth_output_beams = get_output_beams(ground_truth_in, k_in)
images = get_images(ground_truth_output_beams, bins, bandwidth)

for image in images[:5]:
    fig, ax = plt.subplots()
    xx = torch.meshgrid(bins, bins)
    ax.pcolor(xx[0].cpu(), xx[1].cpu(), image.cpu().detach())


# define a unit multivariate normal
from torch.distributions import MultivariateNormal

normal_dist = MultivariateNormal(
    torch.zeros(6), torch.eye(6)
)
normal_samples = normal_dist.sample([10000]).to(**tkwargs)

# define nonparametric transform
from normalizing_flow import NonparametricTransform, track_in_quad

tnf = NonparametricTransform()
tnf.to(tkwargs["device"])

# plot initial beam after transform
p_in_guess = Particle(*tnf(normal_samples).T, **defaults)
plt.hist2d(
    p_in_guess.x.cpu().detach().numpy(), p_in_guess.px.cpu().detach().numpy(), bins=50
)

plt.figure()
p_out_guess = track_in_quad(p_in_guess, k_in[0])
plt.hist2d(
    p_out_guess.x.cpu().detach().numpy(),
    p_out_guess.px.cpu().detach().numpy(),
    bins=50)

# preform optimization with Adam to generate a beam with zero centroid
from normalizing_flow import image_difference_loss, beam_position_loss, zero_centroid_loss

optim = torch.optim.Adam(tnf.parameters(), lr=0.01)
n_iter = 1000
losses = []
image_difference_losses = []
beam_position_losses = []

keys = ["x", "px", "y", "py", "z", "pz"]

true_output_beams = get_output_beams(ground_truth_in, k_in)
true_beam_images = get_images(true_output_beams, bins, bandwidth)

for i in range(n_iter):
    optim.zero_grad()

    #transformed_samples = tnf(normal_samples).double()
    guess_dist = Particle(*tnf(normal_samples).T, **defaults)
    #coords = torch.cat([getattr(guess_dist, key).unsqueeze(0) for key in keys], dim=0)
    #loss = torch.sum(coords.mean(dim=0).pow(2)).sqrt()

    loss = image_difference_loss(
        guess_dist,
        true_beam_images,
        k_in,
        bins=bins,
        bandwidth=bandwidth
    )
    losses += [loss.detach()]
    loss.backward()

    optim.step()

    if i % 100 == 0:
        print(i)

fig, ax = plt.subplots()
ax.semilogy(losses)


plt.show()