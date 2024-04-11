import numpy as np
import matplotlib.pyplot as plt

from pmd_beamphysics.interfaces.bmad import particlegroup_to_bmad
from pmd_beamphysics import ParticleGroup
from bmadx import Particle, M_ELECTRON

def get_stats(h5):
    n_steps = len(h5.keys())
    s = np.zeros(n_steps)
    rms_position = np.zeros((n_steps,3))
    rms_momenta = np.zeros((n_steps,3))
    emittance = np.zeros((n_steps,3))
    geom_emittance = np.zeros((n_steps,3))
    steps = np.zeros((n_steps))

    for i, key in enumerate(h5.keys()):
        steps[i] = h5[key].attrs['Step'][0]
        s[i] = h5[key].attrs['SPOS'][0]
        rms_position[i] = h5[key].attrs['RMSX']
        rms_momenta[i] = h5[key].attrs['RMSP']
        emittance[i] = h5[key].attrs['#varepsilon']
        geom_emittance[i] = h5[key].attrs['#varepsilon-geom']

    sort_ids = np.argsort(steps)
    data = {
        's': s[sort_ids],
        'rms_position': rms_position[sort_ids],
        'rms_momenta': rms_momenta[sort_ids],
        'emittance': emittance[sort_ids],
        'geom_emittance': geom_emittance[sort_ids]
    }

    return data


def plot_stats(stats, sigma_lims=None, epsilon_lims=None, lines=None):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    pos_labels = ['$\sigma_x$', '$\sigma_y$', '$\sigma_z$']
    em_labels = ['$\\varepsilon_x$', '$\\varepsilon_y$', '$\\varepsilon_z$']
    ax[0].plot(stats['s'], stats['rms_position']*1e3, '-', label=pos_labels)
    ax[0].set_ylim(sigma_lims)
    ax[1].plot(stats['s'], stats['emittance']*1e6, '-', label=em_labels)
    ax[1].set_ylim(epsilon_lims)
    if lines is not None:
        for i in range(len(lines)//2):
            ax[0].axvspan(lines[2*i], lines[2*i+1], alpha=0.3, color='grey')
            ax[1].axvspan(lines[2*i], lines[2*i+1], alpha=0.3, color='grey')
    ax[0].legend( loc='lower left')
    ax[1].legend( loc='upper left')
    ax[0].set_ylabel('$\sigma$ [mm]')
    ax[1].set_ylabel('$\\varepsilon$ [mm mrad]')
    ax[1].set_xlabel('s [m]')
    print(stats['emittance'][:,0].min())

    return fig, ax


def last_step(h5):
    last_key_number = len(h5.keys()) - 1
    last_key = 'Step#{}'.format(last_key_number)
    last_step = h5[last_key]

    return last_step


def first_step(h5):
    first_key = 'Step#0'
    last_step = h5[first_key]

    return last_step


def bmad_data_to_bmadx_particle(bmad_data):
    par = Particle(
        x = bmad_data['x'],
        px = bmad_data['px'],
        y = bmad_data['y'],
        py = bmad_data['py'],
        z = bmad_data['z'],
        pz = bmad_data['pz'],
        s = 0.0,
        p0c = bmad_data['p0c'],
        mc2 = M_ELECTRON
    )

    return par


def opal_step_to_bmadx_particle(h5, p0c):
    opal_data = opal_to_data(h5)
    pg = ParticleGroup(data=opal_data)
    pg.drift_to_z(0)
    bmad_data = particlegroup_to_bmad(pg, p0c=p0c)
    bmadx_par = bmad_data_to_bmadx_particle(bmad_data)

    return bmadx_par


def opal_to_data(h5):
    """
    Converts an OPAL step to the standard data format for openPMD-beamphysics
    
    In OPAL, the momenta px, py, pz are gamma*beta_x, gamma*beta_y, gamma*beta_z. 
    
    TODO: More species. 
        
    """
    D = dict(h5.attrs)
    mc2 = D['MASS'][0]*1e9 # GeV -> eV 
    charge = D['CHARGE'][0] # total charge in C
    t = D['TIME'][0] # s
    ptypes = h5['ptype'][:]  # 0 = electron?
    
    # Not used: pref = D['RefPartP']*mc2  # 
    # rref = D['RefPartR']
    
    n = len(ptypes)
    assert all(h5['ptype'][:] == 0)
    species = 'electron'
    status=1
    data = {
        'x':h5['x'][:],
        'y':h5['y'][:],
        'z':h5['z'][:],
        'px':h5['px'][:]*mc2,
        'py':h5['py'][:]*mc2,
        'pz':h5['pz'][:]*mc2,
        't': np.full(n, t),
        'status': np.full(n, status),
        'species':species,
        'weight': np.full(n, abs(charge)/n)
    }

    return data


def save_opal_last_beam(h5, filename):
    out_beam = last_step(h5)
    header = str(len(out_beam['x'][:]))
    data = np.stack([
        out_beam['x'][:],
        out_beam['px'][:], 
        out_beam['y'][:], 
        out_beam['py'][:],
        out_beam['z'][:],
        out_beam['pz'][:]], axis=-1)
    np.savetxt(filename, data, header=header, comments='', delimiter='  ')


def plot_xpx(x, px, x_lims, px_lims):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.hist2d(x*1e3, px*1e3, bins=200)
    ax.set_xlim(x_lims)
    ax.set_ylim(px_lims)
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$p_x/p\ (\\times 10^{3})$')

    return fig, ax


def plot_zpz(z, pz, z_lims, pz_lims):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.hist2d(z*1e3, pz*1e2, bins=200)
    ax.set_xlim(z_lims)
    ax.set_ylim(pz_lims)
    ax.set_xlabel('$z$ [mm]')
    ax.set_ylabel('$\delta$ [%]')

    return fig, ax


def slice_x_px(particle, slices, dim='pz'):
    slice_coord_array = getattr(particle, dim)
    n_slices = len(slices)-1
    slice_x = [None]*n_slices
    slice_px = [None]*n_slices
    slice_ctr = [None]*n_slices
    for i in range(n_slices):
        left = slices[i]
        right = slices[i+1]
        slice_ctr[i] = (right + left) / 2
        indices = np.argwhere((slice_coord_array > left) & (slice_coord_array < right))
        slice_x[i] = particle.x[indices[:,0]]
        slice_px[i] = particle.px[indices[:,0]]

    return slice_x, slice_px, slice_ctr


def get_cov_ellipse(x, px):
    cov = np.cov(x, px)
    lambda_, v = np.linalg.eig(cov)
    a = 2*np.sqrt(lambda_[0])
    b = 2*np.sqrt(lambda_[1])
    avg_x = np.mean(x)
    avg_px = np.mean(px)
    alpha = np.arctan2(v[1,0], v[0,0])
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse = np.array([a*np.cos(theta), b*np.sin(theta)])
    ellipse = np.dot(rotation, ellipse)
    ellipse[0] += avg_x
    ellipse[1] += avg_px

    return ellipse


def plot_sliced_ps_2d(particles, slices, slice_coord, x_lim=None, px_lim=None):
    n_slices = len(slices)-1
    slices_x, slices_px, slices_coords = slice_x_px(particles, slices, dim=slice_coord)
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices))
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i in range(len(slices_x)):
        ax.scatter(
            slices_x[i]*1e3, 
            slices_px[i]*1e3, 
            s=10, 
            color=colors[i], 
            alpha=0.1,
            linewidths=0
        )
        ellipse = get_cov_ellipse(slices_x[i]*1e3, slices_px[i]*1e3)
        ax.plot(
            ellipse[0], 
            ellipse[1], 
            '-',
            linewidth=2,
            color=colors[i], 
            alpha=0.75,
            label=f'${slice_coord}$ = {1e3*slices_coords[i]:.2f} mm',
        )
        ax.set_xlabel('$x$ [mm]')
        ax.set_ylabel('$p_x/p$ ($\\times 10^3$)')
        ax.set_xlim(x_lim)
        ax.set_ylim(px_lim)
        ax.legend(loc='center right', bbox_to_anchor=(1.55, 0.5))

    return fig, ax


def plot_sliced_ps_3d(particles, slices, slice_coord, x_lim=None, px_lim=None, z_lim=None):
    n_slices = len(slices)-1
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(projection='3d')
    slices_x, slices_px, slices_coords = slice_x_px(particles, slices, dim=slice_coord)
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices+2))[1:-1]
    ax.scatter(
        particles.x*1e3, 
        particles.z*1e3,
        particles.px*1e3,
        s=10, 
        cmap='viridis',
        c = particles.z,
        #color=colors[i],
        #color='red', 
        alpha=0.05,
        linewidths=0
    )
    for i in range(len(slices_x)):
        ellipse = get_cov_ellipse(slices_x[i]*1e3, slices_px[i]*1e3)
        ax.plot(
            ellipse[0], 
            np.ones_like(ellipse[0])*slices_coords[i]*1e3,
            ellipse[1],
            '-',
            linewidth=2,
            color=colors[i], 
            alpha=0.75,
            label=f'${slice_coord}$ = {1e3*slices_coords[i]:.2f} mm',
            zorder=100000
        )
        ax.view_init(10, 10, 0)
        ax.set_xlabel('$x$ [mm]')
        ax.set_ylabel('$z$ [mm]')
        ax.set_zlabel('$p_x/p$ ($\\times 10^3$)')
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)
        ax.set_zlim(px_lim)
        #plt.legend()

    return fig,ax

def emittance(par, gamma):
    cov_x = np.cov(par.x, par.px)
    e_x = gamma*np.sqrt(np.linalg.det(cov_x))
    cov_y = np.cov(par.y, par.py)
    e_y = gamma*np.sqrt(np.linalg.det(cov_y))
    cov_z = np.cov(par.z, par.pz)
    e_z = gamma*np.sqrt(np.linalg.det(cov_z))
    return e_x, e_y, e_z