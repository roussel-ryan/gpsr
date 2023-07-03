import torch
from scalene import scalene_profiler

def kde(samples, locations, bandwidth):
    print(f'samples shape: {samples.shape}')
    print(f'locations shape: {locations.shape}')
    print(f'bandwidth: {bandwidth}')

    all_samples = samples.reshape(samples.shape + (1,) * len(locations.shape[:-1]))
    print(f'all_samples shape: {all_samples.shape}')

    samples_copies = all_samples - torch.movedim(locations, -1, 0)
    print(f'samples_copies shape: {samples_copies.shape}')

    diff = torch.norm(
        samples_copies,
        dim=-len(locations.shape[:-1]) - 1,
    )
    print(f'diff shape: {diff.shape}')

    out = (-diff ** 2 / (2.0 * bandwidth ** 2)).exp().sum(dim=len(
                samples.shape)-2)
    print(f'out shape: {out.shape}')

    norm = out.flatten(start_dim=len(locations.shape)-2).sum(dim=-1)
    print(f'norm shape: {norm.shape}')

    result = out / norm.reshape(-1, *(1,)*(len(locations.shape)-1))
    print(f'result shape: {result.shape}')
    return result

bins = torch.linspace(-30, 30, 200) * 1e-3
xx = torch.meshgrid(bins, bins, indexing='ij') 

samples = torch.ones((20, 1000, 2))
locations = torch.stack(xx, dim=-1)
bandwidth = (bins[1]-bins[0]) / 2

# Profiling: 

scalene_profiler.start()

hist = kde(samples, locations, bandwidth)

scalene_profiler.stop()