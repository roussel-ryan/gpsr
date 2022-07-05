import matplotlib.pyplot as plt
import torch

from histogram import histogram2d
from test_beams.noisy_gaussian import generate_beam
from track import Quadrupole, Drift, Lattice

tkwargs = {"device": "cpu", "dtype": torch.float32}
beam = generate_beam(100000, 1e-3, tkwargs)

# get ground truth images
sizes = []
k_in = torch.linspace(-10, 20, 10, **tkwargs).reshape(-1, 1, 1)
bins = torch.linspace(-50, 50, 100, **tkwargs) * 1e-3
bandwidth = torch.tensor(1e-3, **tkwargs)

# define batched lattice
quad = Quadrupole(torch.tensor(0.1, **tkwargs), K1=k_in)
drift = Drift(L=torch.tensor(1.0, **tkwargs))
train_lattice = Lattice([quad, drift], torch)

ground_truth_output_beams = train_lattice(beam)[-1]
images = histogram2d(ground_truth_output_beams.x, ground_truth_output_beams.y, bins,
                     bandwidth)

images = torch.transpose(images, 0, 1)
for i in range(len(images)):
    fig, ax = plt.subplots()
    ax.imshow(images[i, 4])

    

plt.show()

