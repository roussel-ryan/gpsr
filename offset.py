import matplotlib.pyplot as plt
import torch

from histogram import histogram2d
from modeling import ImagingModel, NonparametricTransform, train
from test_beams.noisy_gaussian import generate_beam
from torch_track import TorchQuad, TorchDrift, Lattice
from visualization import compare_images, add_projection

tkwargs = {"device": "cuda", "dtype": torch.float32}
ground_truth_input_beams = generate_beam(100000, 5, 0.0, 2e-3)

# get ground truth images
sizes = []
k_in = torch.linspace(-10, 20, 10).reshape(-1, 1, 1)
bins = torch.linspace(-50, 50, 100) * 1e-3
bandwidth = torch.tensor(1e-3)

# define batched lattice
quad = TorchQuad(torch.tensor(0.1), K1=k_in)
drift = TorchDrift(L=torch.tensor(1.0))
train_lattice = Lattice([quad, drift])

ground_truth_output_beams = train_lattice(ground_truth_input_beams)[-1]

defaults = {
    "s": ground_truth_output_beams.s.float(),
    "p0c": torch.mean(ground_truth_output_beams.pz).float(),
    "mc2": ground_truth_output_beams.mc2.float(),
}

images = histogram2d(
    ground_truth_output_beams.x, ground_truth_output_beams.y, bins, bandwidth
)

train_images = images
# create models
n_models = 5
models = []

# add everything to the gpu w/correct type
bins = bins.to(**tkwargs)
bandwidth = bandwidth.to(**tkwargs)
train_lattice.cuda()
train_images = train_images.cuda()
for name, val in defaults.items():
    defaults[name] = val.to(**tkwargs)

for _ in range(n_models):
    models += [
        ImagingModel(
            NonparametricTransform(),
            bins=bins,
            bandwidth=bandwidth,
            n_particles=10000,
            defaults=defaults,
            n_samples=5
        )
    ]

initial_guess_beams = [model.get_initial_beam(100000) for model in models]

# if needed train the models
fname = "checkpoint.pt"
models = torch.nn.ModuleList(models)
if 0:
    fig, ax = plt.subplots()
    for m in models:
        m.cuda()

        # train the model
        loss = train(
            m,
            train_lattice=train_lattice,
            train_images=train_images,
            n_iter=1000,
            lr=0.1,
        )
        m.cpu()
        torch.cuda.empty_cache()
        ax.semilogy(loss)
    torch.save(models.state_dict(), fname)

models.load_state_dict(torch.load(fname))
predicted_reconstruction = [model.get_initial_beam(100000) for model in models]
models.cuda()

xx = torch.meshgrid(models[0].bins.cpu(), models[0].bins.cpu())

# calculate images
predicted_images = models[0](train_lattice).squeeze()

compare_images(xx, predicted_images[:, 0, ...], train_images[:, 0, ...])

keys = ["x", "px", "y", "py", "z", "pz"]
fig, ax = plt.subplots(len(keys), 1)

gt_beams = ground_truth_input_beams.to_list_of_beams()

for i, val in enumerate(keys):
    add_projection(ax[i], val, predicted_reconstruction, bins=bins/2)
    #add_projection(ax[i], val, initial_guess_beams, bins=bins/2)
    add_projection(ax[i], val, gt_beams, bins=bins/2)

plt.show()
