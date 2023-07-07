import torch
from bmadx import PI, C_LIGHT
from bmadx.bmad_torch.track_torch import TorchDrift, TorchQuadrupole, TorchCrabCavity, TorchSBend, TorchLattice

def quad_drift(l_d = 1.0, l_q = 0.1, n_slices=5):
    '''Creates quad + drift lattice

        Params
        ------ 
            l_d: float
                drift length (m). Default: 1.0

            l_q: float 
                quad length (m). Default: 0.1

            n_steps: int
                slices in quad tracking. Default: 5

        Returns
        -------
            lattice: bmad_torch.TorchLattice
                quad scan lattice
    '''

    q1 = TorchQuadrupole(torch.tensor(l_q),
                         torch.tensor(0.0),
                         n_slices
                         )
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice

def quad_tdc_dipole():
    # Design momentum
    p_design = 10.0e6 # eV/c
    
    # Quadrupole parameters
    l_q = 0.1
    k1 = 0.0 # scan parameter
    
    # Drift from Quad to TDC
    l_d1 = 0.5975
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    #k_tdc = 2.00 # m^-1
    f_tdc = 1.3e9
    #v_tdc = p_design * k_tdc * C_LIGHT / ( 2 * PI * f_tdc )
    v_tdc = 7.0e5
    phi_tdc = 0.0 # scan parameter (maybe?)
    
    # Drift from TDC to Bend
    l_d2 = 0.518
     
    # Bend parameters
    l_bend = 0.365 # arc length
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
                          TILT=torch.tensor(PI/2)
                         )
    
    d2 = TorchDrift(L = torch.tensor(l_d2))
    
    bend = TorchSBend(L = torch.tensor(l_bend),
                      P0C = torch.tensor(p_design),
                      G = torch.tensor(g),
                      E1 = torch.tensor(theta/2),
                      E2 = torch.tensor(theta/2),
                      FRINGE_AT = "no_end"
                     )
    
    d3 = TorchDrift(L = torch.tensor(l_d3))
    
    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])
    
    return lattice