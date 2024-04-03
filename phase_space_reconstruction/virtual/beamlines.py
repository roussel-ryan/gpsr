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


def quad_tdc_bend(p0c, dipole_on=False):
    # Design momentum
    p_design = p0c  # eV/c

    # Quadrupole parameters
    # l_q = 0.08585
    l_q = 0.11
    k1 = 0.0

    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.01
    # k_tdc = 2.00 # m^-1
    f_tdc = 1.3e9
    # v_tdc = p_design * k_tdc * C_LIGHT / ( 2 * PI * f_tdc )
    # v_tdc = 7.0e5
    v_tdc = 0.0  # scan parameter
    phi_tdc = 0.0  # 0-crossing phase

    # Bend parameters
    # fixed:
    l_bend = 0.3018
    # variable when on/off:
    if dipole_on:
        theta = -20.0 * PI / 180.0
        l_arc = l_bend * theta / np.sin(theta)
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC (0.5975)
    l_d1 = 0.790702 - l_q / 2 - l_tdc / 2

    # Drift from TDC to Bend (0.3392)
    l_d2 = 0.631698 - l_tdc / 2 - l_bend / 2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d3 = 0.889 - l_bend / 2 / np.cos(theta)

    # Elements:
    q = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    d1 = TorchDrift(L=torch.tensor(l_d1))

    tdc = TorchCrabCavity(
        L=torch.tensor(l_tdc),
        VOLTAGE=torch.tensor(v_tdc),
        RF_FREQUENCY=torch.tensor(f_tdc),
        PHI0=torch.tensor(phi_tdc),
        TILT=torch.tensor(PI / 2),
    )

    d2 = TorchDrift(L=torch.tensor(l_d2))

    bend = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p_design),
        G=torch.tensor(g),
        # E1 = torch.tensor(theta/2), #double check geometry
        # E2 = torch.tensor(theta/2),
        E1=torch.tensor(0.0),
        E2=torch.tensor(theta),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d3 = TorchDrift(L=torch.tensor(l_d3))

    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])

    return lattice


def quadlet_tdc_bend(p0c, dipole_on=False):
    # Design momentum
    p_design = p0c  # eV/c

    # Quadrupole parameters
    # l_q = 0.08585
    l_q = 0.11
    k1 = 0.0
    l1 = 1.209548 - l_q
    l2 = 0.19685 - l_q
    l3 = 0.18415 - l_q

    # transverse deflecting cavity (TDC) parameters
    l_tdc = 0.01
    # k_tdc = 2.00 # m^-1
    f_tdc = 1.3e9
    # v_tdc = p_design * k_tdc * C_LIGHT / ( 2 * PI * f_tdc )
    # v_tdc = 7.0e5
    v_tdc = 0.0  # scan parameter
    phi_tdc = 0.0  # 0-crossing phase

    # Bend parameters
    # fixed:
    l_bend = 0.3018
    # variable when on/off:
    if dipole_on:
        theta = -20.0 * PI / 180.0
        l_arc = l_bend * theta / np.sin(theta)
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC
    l_d4 = 0.790702 - l_q / 2 - l_tdc / 2

    # Drift from TDC to Bend (0.3392)
    l_d5 = 0.631698 - l_tdc / 2 - l_bend / 2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d6 = 0.889 - l_bend / 2 / np.cos(theta)

    # Elements:
    q1 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    d1 = TorchDrift(L=torch.tensor(l1))

    q2 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    d2 = TorchDrift(L=torch.tensor(l2))

    q3 = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(k1), NUM_STEPS=5)

    d3 = TorchDrift(L=torch.tensor(l3))

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
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p_design),
        G=torch.tensor(g),
        # E1 = torch.tensor(theta/2), #double check geometry
        # E2 = torch.tensor(theta/2),
        E1=torch.tensor(0.0),
        E2=torch.tensor(theta),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d6 = TorchDrift(L=torch.tensor(l_d6))

    lattice = TorchLattice([q1, d1, q2, d2, q3, d3, q4, d4, tdc, d5, bend, d6])

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
