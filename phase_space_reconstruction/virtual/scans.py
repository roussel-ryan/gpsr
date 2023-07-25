import torch
from phase_space_reconstruction.modeling import ImageDataset, ImageDataset2


def run_quad_scan(
        beam,
        lattice,
        screen,
        ks,
        scan_quad_id = 0,
        save_as = None
        ):
    
    """
    Runs virtual quad scan and returns image data from the
    screen downstream.

    Parameters
    ----------
    beam : bmadx.Beam
        ground truth beam
    lattice: bmadx TorchLattice
        diagnostics lattice
    screen: ImageDiagnostic
        diagnostic screen
    ks: Tensor
        quadrupole strengths. 
        shape: n_quad_strengths x n_images_per_quad_strength x 1
    save_as : str
        filename to store output dataset. Default: None.

    Returns
    -------
        dset: ImageDataset
            output image dataset
    """

    # tracking though diagnostics lattice
    diagnostics_lattice = lattice.copy()
    diagnostics_lattice.elements[scan_quad_id].K1.data = ks
    output_beam = diagnostics_lattice(beam)

    # histograms at screen
    images = screen(output_beam)

    # create image dataset
    dset = ImageDataset(ks, images)
    
    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset saved as '{save_as}'")

    return dset

def run_quad_tdc_scan(
        beam,
        lattice,
        screen,
        quad_ks,
        quad_id,
        tdc_vs,
        tdc_id,
        save_as = None
        ):
    
    """
    Runs virtual quad + transverse deflecting cavity 2d scan and returns
    image data from the screen downstream.

    Parameters
    ----------
    beam : bmadx.Beam
        ground truth beam
    lattice: bmadx TorchLattice
        diagnostics lattice
    screen: ImageDiagnostic
        diagnostic screen
    quad_ks: Tensor
        quadrupole strengths. 
        shape: n_quad_strengths
    quad_id: int
        id of quad lattice element used for scan.
    tdc_vs: Tensor
        Transverse deflecting cavity voltages. 
        shape: n_tdc_voltages
    tdc_id: int
        id of tdc lattice element.
    save_as : str
        filename to store output dataset. Default: None.

    Returns
    -------
    dset: ImageDataset
        output image dataset
    """

    # parameter scan mesh
    params = torch.meshgrid(quad_ks, tdc_vs, indexing='ij')

    # tracking though diagnostics lattice
    diagnostics_lattice = lattice.copy()
    diagnostics_lattice.elements[quad_id].K1.data = params[0]
    diagnostics_lattice.elements[tdc_id].VOLTAGE.data = params[1]
    output_beam = diagnostics_lattice(beam)

    # histograms at screen
    images = screen(output_beam)

    # create image dataset
    dset = ImageDataset2(params, images)
    
    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset saved as '{save_as}'")

    return dset