import torch
import matplotlib.pyplot as plt

from phase_space_reconstruction.histogram import histogram2d
from phase_space_reconstruction.beamlines import quad_scan_lattice


def create_data(
        gt_beam,
        quad_strengths,
        save_as = None, 
        verbose = False):
    
    ''' Runs synthetic quad scan and saves image data from the screen
    downstream.

    Args: 
        gt_beam (bmadx.Beam): ground truth beam
        quad_strengths: quadrupole strengths
        folder (str): folder where screen data will be stored
        verbose (bool): if True, plots generated screen images for each
                        quad strength in the scan.

    Returns:
        scan_data: dictionary with screen data

    '''

    tkwargs = {"dtype": torch.float32}
    
    # extra dimension for quad strength tensor
    all_k = torch.tensor(quad_strengths, **tkwargs).unsqueeze(1)
    n_images = len(all_k)

    # tracking though diagnostics lattice
    diagnostics_lattice = quad_scan_lattice()
    diagnostics_lattice.elements[0].K1.data = all_k
    output_beam = diagnostics_lattice(gt_beam)

    # histograms at screen
    bins = torch.linspace(-30, 30, 200, **tkwargs) * 1e-3
    images = []
    bandwidth = (bins[1]-bins[0]) / 2

    for i in range(n_images):
        hist = histogram2d(output_beam.x[i],
                           output_beam.y[i],
                           bins, bandwidth)
        images.append(hist)

    images = torch.cat([ele.unsqueeze(0) for ele in images], dim=0)

    # dict with quad strengths and screen data
    scan_data = {
        'quad_strengths': all_k,
        'images': images,
        'bins': bins
    }
    
    # save scan data if wanted
    if save_as is not None:
        torch.save(scan_data, save_as)

    # plot images if wanted
    if verbose:
        print(f'number of images = {n_images}\n')
        xx = torch.meshgrid(bins, bins, indexing='ij')
        for i in range(0, len(images)):
            print(f'image {i}')
            print(f'k = {all_k[i]} 1/m')
            print(f'stdx = {torch.std(output_beam.x[i])**2 * 1e6} mm')
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pcolor(xx[0]*1e3, xx[1]*1e3, images[i])
            ax.set_xlabel('$x$ (mm)')
            ax.set_ylabel('$y$ (mm)')
            plt.show()

    return scan_data