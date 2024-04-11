import numpy as np
import torch
from bmadx import PI
from bmadx.bmad_torch.track_torch import (
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
    TorchRFCavity,
    TorchSBend,
    TorchSextupole,
)


def quad_drift(l_d=1.0, l_q=0.1, n_slices=5):
    """Creates quad + drift lattice

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
    """

    q1 = TorchQuadrupole(torch.tensor(l_q), torch.tensor(0.0), n_slices)
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice


def sextupole_drift(l_d=1.0, l_q=0.1, n_slices=5):
    """Creates quad + drift lattice

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
    """

    q1 = TorchSextupole(torch.tensor(l_q), torch.tensor(0.0), n_slices)
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice


def quad_tdc_bend(
    p0c,
    l_q = 0.11,
    l_tdc = 0.01,
    f_tdc = 1.3e9,
    phi_tdc = 0.0,
    l_bend = 0.3018,
    theta_on = - 20.0 * PI / 180.0,
    l1 = 0.790702,
    l2 = 0.631698,
    l3 = 0.889,
    dipole_on=False
):
    """
    Creates diagnostic lattice with quad, tdc and bend. 
    Default values are for AWA Zone 5.
    
    Params
    ------
        p0c: float
            design momentum (eV/c). 
            
        l_q: float
            quad length (m). Default: 0.11
        
        l_tdc: float
            TDC length (m). NOTE: for now, Bmad-X TDC is a single kick at the
            TDC center, so this length doesn't change anything. Default: 0.01
        
        f_tdc: float
            TDC frequency (Hz). Default: 1.3e9
        
        phi_tdc: float
            TDC phase (rad). 0.0 corresponds to zero crossing phase with 
            positive slope. Default: 0.0
            
        l_bend: float
            Bend length (m). Default: 0.3018
            
        theta_on: float
            Bending angle when bending magnet is on (rad). Negative angle deflects
            in +x direction. Default: -20*pi/180
        
        l1: float
            Center-to-center distance between quad and TDC (m). Default: 0.790702
        
        l2: float
            Center-to-center distance between TDC and dipole (m). Default: 0.631698
        
        l3: float
            Distance from screens to dipole center (m). Default: 0.889
            
        dipole_on: bool
            Initializes the lattice with dipole on or off. Default: False
            
    Returns
    -------
        TorchLattice
    """
    
    # initialize dipole params when on/off:
    if dipole_on:
        theta = theta_on # negative deflects in +x
        l_arc = l_bend * theta / np.sin(theta)
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC (0.5975)
    l_d1 = l1 - l_q / 2 - l_tdc / 2

    # Drift from TDC to Bend (0.3392)
    l_d2 = l2 - l_tdc / 2 - l_bend / 2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d3 = l3 - l_bend / 2 / np.cos(theta)

    # Elements:
    q = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    d1 = TorchDrift(L=torch.tensor(l_d1))

    tdc = TorchCrabCavity(
        L=torch.tensor(l_tdc),
        VOLTAGE=torch.tensor(0.0),
        RF_FREQUENCY=torch.tensor(f_tdc),
        PHI0=torch.tensor(phi_tdc),
        TILT=torch.tensor(PI / 2),
    )

    d2 = TorchDrift(L=torch.tensor(l_d2))

    bend = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p0c),
        G=torch.tensor(g),
        E1=torch.tensor(0.0),
        E2=torch.tensor(theta),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d3 = TorchDrift(L=torch.tensor(l_d3))

    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])

    return lattice


def quadlet_tdc_bend(
    p0c,
    l_q = 0.11,
    l_tdc = 0.01,
    f_tdc = 1.3e9,
    phi_tdc = 0.0,
    l_bend = 0.3018,
    theta_on = - 20.0 * PI / 180.0,
    l1 = 0.790702,
    l2 = 0.631698,
    l3 = 0.889,
    dipole_on=False,
    lq12 = 1.209548,
    lq23 = 0.19685,
    lq34 = 0.18415,
):
    
    """
    Creates diagnostic lattice with focusing triplet and diagnostics (quad+tdc+bend). 
    Default values are for AWA Zone 5.
    
    Params
    ------
        p0c: float
            design momentum (eV/c). 
            
        l_q: float
            quad length (m). Default: 0.11
        
        l_tdc: float
            TDC length (m). NOTE: for now, Bmad-X TDC is a single kick at the
            TDC center, so this length doesn't change anything. Default: 0.01
        
        f_tdc: float
            TDC frequency (Hz). Default: 1.3e9
        
        phi_tdc: float
            TDC phase (rad). 0.0 corresponds to zero crossing phase with 
            positive slope. Default: 0.0
            
        l_bend: float
            Bend length (m). Default: 0.3018
            
        theta_on: float
            Bending angle when bending magnet is on (rad). Negative angle deflects
            in +x direction. Default: -20*pi/180
        
        l1: float
            Center-to-center distance between quad 4 and TDC (m). Default: 0.790702
        
        l2: float
            Center-to-center distance between TDC and dipole (m). Default: 0.631698
        
        l3: float
            Distance from screens to dipole center (m). Default: 0.889
            
        dipole_on: bool
            Initializes the lattice with dipole on or off. Default: False
            
        lq12: float
            Center-to-center distance between quad 1 and 2 (m). Default: 1.209548
        
        lq23: float
            Center-to-center distance between quad 2 and 3 (m). Default: 0.19685
        
        lq34: float
            Center-to-center distance between quad 3 and 4 (m). Default: 0.18415
            
    Returns
    -------
        TorchLattice
    """
    
    ld1 = lq12 - l_q
    ld2 = lq23 - l_q
    ld3 = lq34 - l_q

    # Elements:
    qq1 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    dd1 = TorchDrift(L=torch.tensor(ld1))

    qq2 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    dd2 = TorchDrift(L=torch.tensor(ld2))

    qq3 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    dd3 = TorchDrift(L=torch.tensor(ld3))
    
    #q_tdc_b = list(quad_tdc_bend(p0c, dipole_on).elements)
    #print(q_tdc_b)
    #print([q1, d1, q2, d2, q3, d3] + q_tdc_b)
    
    #lattice = TorchLattice([q1, d1, q2, d2, q3, d3] + q_tdc_b)
    
    #################
    
    # initialize dipole params when on/off:
    if dipole_on:
        theta = theta_on # negative deflects in +x
        l_arc = l_bend * theta / np.sin(theta)
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC (0.5975)
    l_d1 = l1 - l_q / 2 - l_tdc / 2

    # Drift from TDC to Bend (0.3392)
    l_d2 = l2 - l_tdc / 2 - l_bend / 2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d3 = l3 - l_bend / 2 / np.cos(theta)

    # Elements:
    q = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    d1 = TorchDrift(L=torch.tensor(l_d1))

    tdc = TorchCrabCavity(
        L=torch.tensor(l_tdc),
        VOLTAGE=torch.tensor(0.0),
        RF_FREQUENCY=torch.tensor(f_tdc),
        PHI0=torch.tensor(phi_tdc),
        TILT=torch.tensor(PI / 2),
    )

    d2 = TorchDrift(L=torch.tensor(l_d2))

    bend = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p0c),
        G=torch.tensor(g),
        E1=torch.tensor(0.0),
        E2=torch.tensor(theta),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d3 = TorchDrift(L=torch.tensor(l_d3))
    
    ###############
    
    lattice = TorchLattice([qq1, dd1, qq2, dd2, qq3, dd3, q, d1, tdc, d2, bend, d3])
        
    return lattice


def facet_ii_SC20(p0c, dipole_on=False):
    # Design momentum
    p_design = p0c  # eV/c

    # Quadrupole parameters
    # l_q = 0.08585
    l_q = 0.714
    k1 = 0.0

    # transverse deflecting cavity (TDC) parameters
    l_tdc = 1.0334
    f_tdc = 1.1424e10
    v_tdc = 0.0  # scan parameter
    phi_tdc = 0.0  # 0-crossing phase

    # Bend parameters
    # fixed:
    l_bend = 0.9779
    # variable when on/off:
    if dipole_on:
        g = -6.1356e-3
        e1 = 3e-3
        e2 = 3e-3

    if not dipole_on:
        g = 2.22e-16  # machine epsilon to avoid numerical error
        e1 = 0.0
        e2 = 0.0

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC
    l_d4 = 3.464

    # Drift from TDC to Bend (0.3392)
    l_d5 = 19.223

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d6 = 8.8313

    # Elements:
    #q1 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    #d1 = TorchDrift(L=torch.tensor(l1))

    #q2 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    #d2 = TorchDrift(L=torch.tensor(l2))

    #q3 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    #d3 = TorchDrift(L=torch.tensor(l3))

    q4 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    d4 = TorchDrift(L=torch.tensor(l_d4))

    tdc = TorchCrabCavity(
        L=torch.tensor(l_tdc),
        VOLTAGE=torch.tensor(v_tdc),
        RF_FREQUENCY=torch.tensor(f_tdc),
        PHI0=torch.tensor(phi_tdc),
        TILT=torch.tensor(PI / 2),
    )

    d5 = TorchDrift(L=torch.tensor(l_d5))

    bend = TorchSBend(
        L=torch.tensor(l_bend),
        P0C=torch.tensor(p_design),
        G=torch.tensor(g),
        # E1 = torch.tensor(theta/2), #double check geometry
        # E2 = torch.tensor(theta/2),
        E1=torch.tensor(e1),
        E2=torch.tensor(e2),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d6 = TorchDrift(L=torch.tensor(l_d6))

    #lattice = TorchLattice([q1, d1, q2, d2, q3, d3, q4, d4, tdc, d5, bend, d6])
    lattice = TorchLattice([q4, d4, tdc, d5, bend, d6])

    return lattice
