import sys
import os
import torch
from phase_space_reconstruction.train import train_3d_scan

seed = sys.argv[1]
gpu_num = sys.argv[2]

gt = torch.load(os.path.join('data','non_gaussian_beam','stats','non_gaussian_beam.pt'))
train_dset = torch.load(os.path.join('data','non_gaussian_beam','stats','3d_scan_train.dset'))

# reference momentum in eV/c
p0c = 43.36e6 

# triplet params obtained from triplet-opt.ipynb
k1 = 7.570125
k2 = -15.704693
k3 = 1.0

# diagnostic beamline:
lattice = quadlet_tdc_bend(p0c=p0c, dipole_on=False)
lattice.elements[0].K1.data = torch.tensor(k1)
lattice.elements[2].K1.data = torch.tensor(k2)
lattice.elements[4].K1.data = torch.tensor(k3)

# Scan over quad strength, tdc on/off and dipole on/off
scan_ids = [6, 8, 10] 
n_ks = 5
ks = torch.linspace(-10, 10, n_ks) # quad ks
vs = torch.tensor([0, 5e6]) # TDC off/on
gs = torch.tensor([2.22e-16, 20.0*PI/180.0/0.365]) # dipole off/on
train_params = torch.stack(torch.meshgrid(ks, vs, gs, indexing='ij'))

# create diagnostic screen: 
bins = torch.linspace(-50, 50, 200) * 1e-3
bandwidth = (bins[1]-bins[0]) / 2
screen = ImageDiagnostic(bins, bins, bandwidth)

fname = f'r_{seed}.pt'
torch.manual_seed(seed)
pred_beam = train_3d_scan(
    train_dset, 
    lattice, 
    p0c, 
    screen,
    ids = scan_ids,
    n_epochs = 1_000, 
    n_particles = 100_000, 
    device = torch.device(f'cuda:{gpu_num}')
    save_as = os.path.join('data','non_gaussian_beam','stats','fname')
)