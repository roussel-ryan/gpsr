import torch
from distgen import Generator
from distgen.physical_constants import unit_registry as unit
from matplotlib import pyplot as plt

from phase_space_reconstruction.histogram import histogram2d
from bmadx.bmad_torch.track_torch import Beam

from beamline import create_quad_scan_beamline

def generate_test_beam():
    k = 2 * 3.14 / (30 * unit("mm"))

    pycos = {
        "type": "cosine y:py",
        "amplitude": {"value": 0.05, "units": "MeV/c"},
        "omega": {"value": k.magnitude, "units": str(k.units)},
        "phase": {"value": 90, "units": "deg"},
    }

    linear_position = {
        "type": "polynomial x:y",
        "coefficients": [
            {"value": -0.005, "units": "m"},
            {"value": 0.75, "units": ""},
            {"value": 50.0, "units":"1/m"}
        ]
    }

    gen = Generator("gaussian.yaml")

    twiss_x = {
        "type": "set_twiss x",
        "beta": {
            "value": 9,
            "units": "m",
        },
        "alpha": {"value": 5, "units": ""},
        "emittance": {"value": 2.0, "units": "um"},
    }

    twiss_y = {
        "type": "set_twiss y",
        "beta": {
            "value": 9,
            "units": "m",
        },
        "alpha": {"value": 0, "units": ""},
        "emittance": {"value": 2.0, "units": "um"},
    }

    gen.input["transforms"] = {
        "twissx": twiss_x,
        "twissy": twiss_y,
        "pycos": pycos,
        #"linear_energy": linear_energy,
        "linear_position": linear_position,
        "order": ["twissx","twissy", "linear_position", "pycos"],#, "linear_energy"],
    }
    gen.run()
    particles_3 = gen.particles

    particles = particles_3

    particles.plot("x", "y")
    particles.plot("x", "px")
    particles.plot("y", "py")
    particles.plot("x", "py")
    particles.plot("y", "px")
    particles.plot("z", "pz")

    # dump test beam to pytorch file
    keys = ["x", "px", "y", "py", "z", "pz"]
    data = torch.cat(
        [torch.tensor(getattr(particles, key)).unsqueeze(0) for key in keys]
    ).T

    p0c = torch.mean(data[:, -1])

    data[:, 1] = data[:, 1] / p0c
    data[:, 3] = data[:, 3] / p0c
    data[:, -1] = (data[:, -1] - p0c) / p0c

    torch.save(data, "ground_truth_dist.pt")


def generate_test_images():
    tkwargs = {"device": "cuda", "dtype": torch.float32}

    generate_test_beam()

    # load gt beam
    beam_coords = torch.load("ground_truth_dist.pt")
    input_beam = Beam(
        beam_coords,
        s=torch.tensor(0.0, **tkwargs),
        p0c=torch.mean(beam_coords[:, -1]),
        mc2=torch.tensor(0.511e6, **tkwargs),
    )

    n_images = 20
    k_in = torch.linspace(-25, 15, n_images, **tkwargs).unsqueeze(1)
    bins = torch.linspace(-30, 30, 50, **tkwargs) * 1e-3

    # create synthetic images
    train_lattice = create_quad_scan_beamline()
    train_lattice.elements[0].K1.data = k_in

    train_lattice = train_lattice.cuda()
    input_beam = input_beam.cuda()

    output_beam = train_lattice(input_beam)
    screen_data = torch.cat(
        [ele.unsqueeze(-1) for ele in [output_beam.x, output_beam.y]], dim=-1
    ).cpu().float()

    # do histogramming
    images = []
    bins = bins.cpu()
    bin_width = bins[1]-bins[0]
    bandwidth = bin_width.cpu() / 2
    for i in range(n_images):
        hist = histogram2d(screen_data[i].T[0], screen_data[i].T[1], bins, bandwidth)
        images.append(hist)

    images = torch.cat([ele.unsqueeze(0) for ele in images], dim=0)

    for i in range(0, len(images), 3):
        plt.figure()
        plt.imshow(images[i])

        plt.figure()
        plt.hist(output_beam.x[i].detach().cpu().numpy())
        plt.hist(output_beam.y[i].detach().cpu().numpy())

        print(k_in[i], torch.std(output_beam.x[i])**2 * 1e6)

    xx = torch.meshgrid(bins, bins)
    # save image data
    torch.save(images.unsqueeze(1), "train_images.pt")
    torch.save(k_in, "kappa.pt")
    torch.save(bins, "bins.pt")
    torch.save(xx, "xx.pt")


if __name__=="__main__":
    generate_test_images()
    plt.show()