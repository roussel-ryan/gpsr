import torch
from phase_space_reconstruction.modeling import ImageDataset


def run_quad_scan(
        beam,
        lattice,
        screen,
        ks,
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
    diagnostics_lattice = lattice
    diagnostics_lattice.elements[0].K1.data = ks
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