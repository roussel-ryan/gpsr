import torch
from bmadx import PI
from bmadx.bmad_torch.track_torch import TorchDrift, TorchQuadrupole, TorchCrabCavity, TorchRFCavity, TorchSBend, TorchLattice

def create_quad_scan_beamline():
    q1 = TorchQuadrupole(torch.tensor(0.1), torch.tensor(0.0), 5)
    d1 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([q1, d1])
    return lattice

def create_6d_diagnostic_beamline():
    # Design momentum
    p_design = 4.0e7 # eV/c
    
    # Quadrupole parameters
    l_q = 1e-3
    k1 = 0.0 # scan parameter
    
    # Drift from Quad to TDC
    l_d1 = 0.165 + 0.105 + 0.165 + 0.3275 - 0.23/2
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    v_tdc = 1e4
    f_tdc = 1.3e9
    phi_tdc = 0.0 #scan parameter (maybe?)
    
    # Drift from TDC to Bend
    l_d2 = 0.633 - 0.23/2
     
    # Bend parameters
    l_bend = 0.365 # arc length
    p_bend = 4e7 # reference momentum
    theta = 20.0 * PI / 180.0 # angle
    g = theta/l_bend # curvature function. positive bends in the -x direction. 
    
    # Drift from Bend to YAG
    l_d3 = 0.895
    
    # Elements:
    q = TorchQuadrupole(L = torch.tensor(l_q),
                        K1 = torch.tensor(k1),
                        NUM_STEPS = 5)
    
    d1 = TorchDrift(L = torch.tensor(l_d1))
    
    tdc = TorchCrabCavity(L = torch.tensor(l_tdc),
                          VOLTAGE = torch.tensor(v_tdc),
                          RF_FREQUENCY = torch.tensor(f_tdc),
                          PHI0 = torch.tensor(phi_tdc),
                          P0C = torch.tensor(p_bend),
                          G = torch.tensor(g)
                         )
    
    d2 = TorchDrift(L = torch.tensor(l_d2))
    
    bend = TorchSBend(L = torch.tensor(l_bend),
                      P0C = torch.tensor(p_design),
                      G = torch.tensor(g),
                      E1 = torch.tensor(theta/2),
                      E2 = torch.tensor(theta/2)
                     )
    
    d3 = TorchDrift(L = torch.tensor(l_d3))
    
    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])
    
    return lattice