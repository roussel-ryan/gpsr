import torch
import numpy as np
from bmadx import PI
from bmadx.bmad_torch.track_torch import (
    TorchDrift,
    TorchQuadrupole,
    TorchCrabCavity,
    TorchRFCavity,
    TorchSBend,
    TorchLattice, TorchSextupole
)


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

def sextupole_drift(l_d = 1.0, l_q = 0.1, n_slices=5):
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

    q1 = TorchSextupole(torch.tensor(l_q),
                         torch.tensor(0.0),
                         n_slices
                         )
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice

def quad_tdc_bend(p0c, dipole_on=False):

    # Design momentum
    p_design = p0c # eV/c
    
    # Quadrupole parameters
    l_q = 0.1
    k1 = 0.0
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    #k_tdc = 2.00 # m^-1
    f_tdc = 1.3e9
    #v_tdc = p_design * k_tdc * C_LIGHT / ( 2 * PI * f_tdc )
    #v_tdc = 7.0e5
    v_tdc = 0.0 # scan parameter
    phi_tdc = 0.0 # 0-crossing phase
     
    # Bend parameters
    # fixed: 
    l_bend = 0.365
    # variable when on/off: 
    if dipole_on:
        theta = 20.0 * PI / 180.0
        l_arc = l_bend * theta / (2 * np.sin(theta/2))
        g = theta / l_arc 
    if not dipole_on:
        g = 2.22e-16 # machine epsilon to avoid numerical error
        theta = 2*np.arcsin(l_bend*g/2)
        l_arc = theta/g
    
    # Drifts with geometrical corrections: 

    # Drift from Quad to TDC (0.5975)
    l_d1 = 0.7625 - l_q/2 - l_tdc/2

    # Drift from TDC to Bend (0.3392)
    l_d2 = 0.633 - l_tdc/2 - l_bend/2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d3 = 0.895 - l_bend/2/np.cos(theta)
    
    # Elements:
    q = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
    )
    
    d1 = TorchDrift(L = torch.tensor(l_d1))
    
    tdc = TorchCrabCavity(
        L = torch.tensor(l_tdc),
        VOLTAGE = torch.tensor(v_tdc),
        RF_FREQUENCY = torch.tensor(f_tdc),
        PHI0 = torch.tensor(phi_tdc),
        TILT = torch.tensor(PI/2)
    )
    
    d2 = TorchDrift(L = torch.tensor(l_d2))

    bend = TorchSBend(
        L = torch.tensor(l_arc),
        P0C = torch.tensor(p_design),
        G = torch.tensor(g),
        E1 = torch.tensor(theta/2),
        E2 = torch.tensor(theta/2),
        FRINGE_AT = "no_end"
    )

    d3 = TorchDrift(L = torch.tensor(l_d3))
    
    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])
    
    return lattice

def triplet_tdc_bend(p0c):
    # Design momentum
    p_design = p0c # eV/c
    
    # Quadrupole parameters
    l_q = 0.1
    k1 = 0.0 
    d_q = 0.1 #distance between quads in triplet
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    f_tdc = 1.3e9
    phi_tdc = 0.0 # zero-crossing phase
    v_tdc = 0.0 # scan parameter
     
    # Bend parameters
    l_arc = 0.365 # arc length
    theta = 20.0 * PI / 180.0 # angle in radians
    g = theta / l_arc # curvature function. positive bends in the -x direction. 
    l_bend = l_arc / theta * np.sin(theta)
    
    # Drifts with geometrical corrections: 

    # Drift from Quad to TDC
    l_d3 = 0.7625 - l_q/2 - l_tdc/2

    # Drift from TDC to Bend
    l_d4 = 0.633 - l_tdc/2 - l_bend/2

    # Drift from Bend to YAG 2 (bend on)
    l_d5 = 0.895 - l_bend/2/np.cos(theta)
    
    # Elements:
    q1 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d1 = TorchDrift( L = torch.tensor(d_q) )

    q2 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d2 = TorchDrift( L = torch.tensor(d_q) )

    q3 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d3 = TorchDrift( L = torch.tensor(l_d3) )
    
    tdc = TorchCrabCavity(
        L = torch.tensor(l_tdc),
        VOLTAGE = torch.tensor(v_tdc),
        RF_FREQUENCY = torch.tensor(f_tdc),
        PHI0 = torch.tensor(phi_tdc),
        TILT=torch.tensor(PI/2)
        )
    
    d4 = TorchDrift( L = torch.tensor(l_d4) )

    bend = TorchSBend(
        L = torch.tensor(l_arc),
        P0C = torch.tensor(p_design),
        G = torch.tensor(g),
        E1 = torch.tensor(theta/2),
        E2 = torch.tensor(theta/2),
        FRINGE_AT = "no_end"
        )

    d5 = TorchDrift( L = torch.tensor(l_d5) )
    
    lattice = TorchLattice(
        [q1, d1, q2, d2, q3, d3, tdc, d4, bend, d5]
        )
    
    return lattice

def quadlet_tdc_bend(p0c, dipole_on = False):
    # Design momentum
    p_design = p0c # eV/c
    
    # Quadrupole parameters
    l_q = 0.1
    k1 = 0.0 
    d_q = 0.1 #distance between quads in triplet
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    f_tdc = 1.3e9
    phi_tdc = 0.0 # zero-crossing phase
    v_tdc = 0.0 # scan parameter
     
    # Bend parameters
    # fixed: 
    l_bend = 0.365
    # variable when on/off: 
    if dipole_on:
        theta = 20.0 * PI / 180.0
        l_arc = l_bend * theta / (2 * np.sin(theta/2))
        g = theta / l_arc 
    if not dipole_on:
        g = 2.22e-16 # machine epsilon to avoid numerical error
        theta = 2*np.arcsin(l_bend*g/2)
        l_arc = theta/g

    # Drift from Quad to TDC
    l_d4 = 0.7625 - l_q/2 - l_tdc/2

    # Drift from TDC to Bend
    l_d5 = 0.633 - l_tdc/2 - l_bend/2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d6 = 0.895 - l_bend/2/np.cos(theta)
    
    # Elements:
    q1 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d1 = TorchDrift( L = torch.tensor(d_q) )

    q2 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d2 = TorchDrift( L = torch.tensor(d_q) )

    q3 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d3 = TorchDrift( L = torch.tensor(d_q) )

    q4 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d4 = TorchDrift( L = torch.tensor(l_d4) )
    
    tdc = TorchCrabCavity(
        L = torch.tensor(l_tdc),
        VOLTAGE = torch.tensor(v_tdc),
        RF_FREQUENCY = torch.tensor(f_tdc),
        PHI0 = torch.tensor(phi_tdc),
        TILT=torch.tensor(PI/2)
        )
    
    d5 = TorchDrift( L = torch.tensor(l_d5) )

    bend = TorchSBend(
        L = torch.tensor(l_arc),
        P0C = torch.tensor(p_design),
        G = torch.tensor(g),
        E1 = torch.tensor(theta/2),
        E2 = torch.tensor(theta/2),
        FRINGE_AT = "no_end"
        )

    d6 = TorchDrift( L = torch.tensor(l_d6) )
    
    lattice = TorchLattice(
        [q1, d1, q2, d2, q3, d3, q4, d4, tdc, d5, bend, d6]
        )
    
    return lattice

def quadlet_quad_tdc_bend(p0c, dipole_on = False):
    # Design momentum
    p_design = p0c # eV/c
    
    # Quadrupole parameters
    l_q = 0.1
    k1 = 0.0 
    d_q = 0.1 #distance between quads in quadlet
    
    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.23
    f_tdc = 1.3e9
    phi_tdc = 0.0 # zero-crossing phase
    v_tdc = 0.0 # scan parameter
     
    # Bend parameters
    # fixed: 
    l_bend = 0.365
    # variable when on/off: 
    if dipole_on:
        theta = 20.0 * PI / 180.0
        l_arc = l_bend * theta / (2 * np.sin(theta/2))
        g = theta / l_arc 
    if not dipole_on:
        g = 2.22e-16 # machine epsilon to avoid numerical error
        theta = 2*np.arcsin(l_bend*g/2)
        l_arc = theta/g

    # Drift from quad 4 to scanning quad
    l_d4 = 0.1

    # Drift from Quad to TDC
    l_d5 = 0.7625 - l_q/2 - l_tdc/2

    # Drift from TDC to Bend
    l_d6 = 0.633 - l_tdc/2 - l_bend/2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d7 = 0.895 - l_bend/2/np.cos(theta)
    
    # Elements:
    q1 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d1 = TorchDrift( L = torch.tensor(d_q) )

    q2 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d2 = TorchDrift( L = torch.tensor(d_q) )

    q3 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d3 = TorchDrift( L = torch.tensor(d_q) )

    q4 = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d4 = TorchDrift( L = torch.tensor(l_d4) )

    q_scan = TorchQuadrupole(
        L = torch.tensor(l_q),
        K1 = torch.tensor(k1),
        NUM_STEPS = 5
        )
    
    d5 = TorchDrift( L = torch.tensor(l_d5) )
    
    tdc = TorchCrabCavity(
        L = torch.tensor(l_tdc),
        VOLTAGE = torch.tensor(v_tdc),
        RF_FREQUENCY = torch.tensor(f_tdc),
        PHI0 = torch.tensor(phi_tdc),
        TILT=torch.tensor(PI/2)
        )
    
    d6 = TorchDrift( L = torch.tensor(l_d6) )

    bend = TorchSBend(
        L = torch.tensor(l_arc),
        P0C = torch.tensor(p_design),
        G = torch.tensor(g),
        E1 = torch.tensor(theta/2),
        E2 = torch.tensor(theta/2),
        FRINGE_AT = "no_end"
        )

    d7 = TorchDrift( L = torch.tensor(l_d7) )
    
    lattice = TorchLattice(
        [q1, d1, q2, d2, q3, d3, q4, d4, q_scan, d5, tdc, d6, bend, d7]
        )
    
    return lattice

def test_beamline(p0c):
    l_drift = torch.tensor(1.0)
    l_quad = torch.tensor(0.1)
    d_quad = torch.tensor(0.1)
    l_cav = torch.tensor(0.2)
    l_bend = torch.tensor(0.3)
    k = torch.tensor(0.0)
    k_skew = torch.tensor(5.0)
    rf_freq = torch.tensor(1.3e9)
    voltage = torch.tensor(1e7)

    q1 = TorchQuadrupole(
        L = l_quad,
        K1 = k,
        NUM_STEPS=5
    )
    d1 = TorchDrift(L = d_quad)
    q2 = TorchQuadrupole(
        L = l_quad,
        K1 = k,
        NUM_STEPS=5
    )
    d2 = TorchDrift(L = d_quad)
    q3 = TorchQuadrupole(
        L = l_quad,
        K1 = k,
        NUM_STEPS=5
    )
    d3 = TorchDrift(L = d_quad)
    q_skew = TorchQuadrupole(
        L = l_quad,
        K1 = k_skew,
        NUM_STEPS=5,
        TILT = torch.tensor(3*PI/4)
    )
    d4 = TorchDrift(L = l_drift)
    rf_cav = TorchRFCavity(
        L = l_cav,
        VOLTAGE = voltage,
        RF_FREQUENCY = rf_freq 
    )
    d5 = TorchDrift(L = l_drift)
    crab_cav = TorchCrabCavity(
        L = l_cav,
        VOLTAGE = -voltage,
        RF_FREQUENCY = rf_freq,
        TILT=torch.tensor(PI/2)
        )
    
    d6 = TorchDrift( L = l_drift )
    l_arc = 0.365 # arc length
    theta = 20.0 * PI / 180.0 # angle in radians
    g = theta / l_arc # curvature function. positive bends in the -x direction. 
    l_bend = l_arc / theta * np.sin(theta) # 0.3576
    bend = TorchSBend(
        L = torch.tensor(l_arc),
        P0C = torch.tensor(p0c),
        G = torch.tensor(g),
        E1 = torch.tensor(0.0),
        E2 = torch.tensor(theta),
        FRINGE_AT = "no_end"
        )
    d7 = TorchDrift(L = l_drift)

    lattice = TorchLattice(
        [q1, d1, q2, d2, q3, d3, q_skew, d4, rf_cav, d5, crab_cav, d6, bend, d7]
        )
    
    return lattice

