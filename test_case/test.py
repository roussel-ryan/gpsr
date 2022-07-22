import matplotlib.pyplot as plt
import numpy as np
import torch
from modeling import Imager, InitialBeam, QuadScanTransport
from torch.utils.data import DataLoader, Dataset, random_split
from torchensemble import VotingRegressor

all_k = torch.load("kappa.pt")
all_images = torch.load("images.pt").unsqueeze(1)
bins = torch.load("bins.pt")

bins = (bins[:-1] + bins[1:]) / 2
xx = torch.meshgrid(bins, bins)

all_k = all_k.cpu()
all_images = all_images.cpu()


# create data loader
class ImageDataset(Dataset):
    def __init__(self, k, images):
        self.images = images
        self.k = k

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.k[idx], self.images[idx]


class QuadScanModel(torch.nn.Module):
    def __init__(self, initial_beam, transport, imager):
        super(QuadScanModel, self).__init__()
        self.beam_generator = initial_beam
        self.lattice = transport
        self.imager = imager

    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)
        return output_images


train_dset = torch.load("train.dset")
test_dset = torch.load("test.dset")

train_dataloader = DataLoader(train_dset, batch_size=2)
test_dataloader = DataLoader(test_dset)


bandwidth = torch.tensor(1.0e-4)

defaults = {
    "s": torch.tensor(0.0).float(),
    "p0c": torch.tensor(10.0e6).float(),
    "mc2": torch.tensor(0.511e6).float(),
}

module_kwargs = {
    "initial_beam": InitialBeam(100000, **defaults),
    "transport": QuadScanTransport(torch.tensor(0.1), torch.tensor(1.0)),
    "imager": Imager(bins, bandwidth),
}


ensemble = VotingRegressor(
    estimator=QuadScanModel, estimator_args=module_kwargs, n_estimators=5, n_jobs=1
)


criterion = torch.nn.MSELoss(reduction="sum")
ensemble.set_criterion(criterion)

ensemble.set_optimizer("Adam", lr=0.01)

from torchensemble.utils import io

io.load(ensemble, ".")
ensemble.cpu()

train_k = all_k[train_dset.indices]
train_k = train_k.cpu()

train_im = all_images[train_dset.indices]
train_im = train_im.cpu()

test_k = all_k[test_dset.indices]
test_k = test_k.cpu()

test_im = all_images[test_dset.indices]
test_im = test_im.cpu()
print(all_images.shape)

recompute_images = False
with torch.no_grad():
    if recompute_images:
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        custom_beam = None  # dist.sample([100000]).cuda()

        model_pred = torch.cat(
            [ensemble[i](all_k[:, :1]).unsqueeze(0) for i in range(len(ensemble))]
        )
        model_pred = torch.transpose(model_pred.squeeze(dim=2), 0, 1)
        model_pred = model_pred.cpu().detach()

        torch.save(model_pred, "all_pred_images.pt")
    else:
        model_pred = torch.load("all_pred_images.pt")

# calculate moments of a single image
train_image = all_images[-1, 0]
test_image = model_pred[-1, 0]


def f(image, _xx):
    fig, ax = plt.subplots()
    c = ax.pcolor(*_xx, image)
    fig.colorbar(c)

    x = _xx[0].T[0]
    y = _xx[1][0]

    # get proj
    proj_x = image.sum(dim=-1)
    proj_y = image.sum(dim=-2)

    mean_x = torch.sum(x * proj_x) / torch.sum(proj_x)
    mean_y = torch.sum(y * proj_y) / torch.sum(proj_y)

    var_x = torch.sum(proj_x*(x - mean_x)**2) / torch.sum(proj_x)
    var_y = torch.sum(proj_y*(y - mean_y)**2) / torch.sum(proj_y)
    print(var_x, var_y)

    print(np.cov(y.numpy(), aweights=proj_y.numpy()))

    return mean_y, var_y, proj_y


x = xx[0].T[0]
m, v, proj_x = f(train_image, xx)
pm, pv, pproj_x = f(test_image, xx)

dist = torch.distributions.Normal(m, v.sqrt())
pdist = torch.distributions.Normal(pm, pv.sqrt())

print(dist)
print(pdist)
fig,ax = plt.subplots()
ax.plot(x, proj_x)
ax.plot(x, dist.log_prob(x).exp() / dist.log_prob(x).exp().sum())

fig2,ax2 = plt.subplots()
ax2.plot(x, pproj_x)
ax2.plot(x, pdist.log_prob(x).exp() / pdist.log_prob(x).exp().sum())

fig2,ax2 = plt.subplots()
ax2.plot(x, proj_x)
ax2.plot(x, pproj_x)

plt.show()
