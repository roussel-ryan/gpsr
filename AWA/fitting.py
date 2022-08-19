import logging

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset
from torchensemble import VotingRegressor, SnapshotEnsembleRegressor

import sys
sys.path.append("../")

from modeling import Imager, QuadScanTransport, ImageDataset, \
    QuadScanModel, InitialBeam, \
    NonparametricTransform

logging.basicConfig(level=logging.INFO)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)


def det(A):
    return A[0, 0] * A[1, 1] - A[1, 0] ** 2

class MaxEntropyQuadScan(QuadScanModel):
    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)
        # return output_images

        #scalar_metric = 0
        # calculate 6D emittance of input beam
        # emit =
        cov = torch.cov(initial_beam.data.T) + torch.eye(6, device = initial_beam.data.device)*1e-8
        #self.covs.append(cov)
        exp_factor = torch.det(2*3.14*2.71*cov)
        #print(exp_factor)
        #scalar_metric = torch.norm(initial_beam.data, dim=1).pow(2).mean()

        
        return output_images, -0.5*torch.log(exp_factor)

def kl_div(target, pred):
    eps = 1e-8
    return target * ((target + eps).log() - (pred + eps).log())
    
def weighted_mse_loss(target, pred):
    return (target - pred)**2
    
class CustomLoss(torch.nn.MSELoss):
    def __init__(self):
        super().__init__()
        self.loss_record = []
        
    def forward(self, input_data, target):
        image_loss = weighted_mse_loss(target, input_data[0]).sum()
        # return image_loss + 1.0 * input[1]
        eps = 1e-8
        #image_loss = torch.sum(kl_div(targe, input_data[0]))
        #print(image_loss, 1e-4*input_data[1])
        
        entropy_loss = 1e-4*input_data[1]
        self.loss_record.append([image_loss, entropy_loss])
        return image_loss + entropy_loss


def create_ensemble(bins, bandwidth):
    defaults = {
        "s": torch.tensor(0.0).float(),
        "p0c": torch.tensor(65.0e6).float(),
        "mc2": torch.tensor(0.511e6).float(),
    }

    transformer = NonparametricTransform(4, 50, 0.0, torch.nn.Tanh())
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
    
    module_kwargs = {
        "initial_beam": InitialBeam(100000, transformer, base_dist, **defaults),
        "transport": QuadScanTransport(torch.tensor(0.12), torch.tensor(2.84 + 0.54), 1),
        "imager": Imager(bins, bandwidth),
        "condition": False
    }

    ensemble = VotingRegressor(
        estimator=MaxEntropyQuadScan, 
        estimator_args=module_kwargs, 
        n_estimators=5
    )
    return ensemble

def get_data():
    folder = ""
    all_k = torch.load(folder + "kappa.pt")
    all_images = torch.load(folder + "train_images.pt")
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0]

    all_k = all_k.cuda()[:,:1]
    all_images = all_images.cuda()[:,:1]
    
    return all_k, all_images, bins
    
    
if __name__ == "__main__":
    
    all_k, all_images, bins = get_data()
    print(all_k.shape)
    print(all_images.shape)

    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])
    #split = 18
    #train_dset = Subset(dset, range(split))
    #test_dset = Subset(dset, range(split, len(dset)))
    
    #train_dset, test_dset = random_split(
    #    dset, [16, 4]
    #)
    
    train_dataloader = DataLoader(train_dset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    torch.save(train_dset, "train.dset")
    torch.save(test_dset, "test.dset")

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width/2
    print(bandwidth)

    ensemble = create_ensemble(bins, bandwidth)
    
    # criterion = torch.nn.MSELoss(reduction="sum")
    criterion = CustomLoss()
    ensemble.set_criterion(criterion)

    n_epochs = 400
    #ensemble.set_scheduler("CosineAnnealingLR", T_max=n_epochs)
    ensemble.set_scheduler("StepLR", gamma=0.1, step_size=200, verbose=False)
    #ensemble.set_scheduler("CosineAnnealingLR", T_max=n_epochs)
    ensemble.set_optimizer("Adam", lr=0.01)
    
    ensemble.fit(train_dataloader, epochs=n_epochs, save_dir="alpha_1e-4")#, test_loader=test_dataloader)#,
    # lr_clip=[0.005,
    # 0.01])


