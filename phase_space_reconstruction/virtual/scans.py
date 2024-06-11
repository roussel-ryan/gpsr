import torch

from phase_space_reconstruction.modeling import ImageDataset, ImageDataset3D


def run_quad_scan(beam, lattice, screen, ks, scan_quad_id=0, save_as=None):
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


def run_sextupole_scan(beam, lattice, screen, ks, scan_quad_id=0, save_as=None):
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
    diagnostics_lattice.elements[scan_quad_id].K2.data = ks
    print(list(diagnostics_lattice.named_parameters()))
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


def run_awa_3d_scan(beam, lattice, screen, ks, vs, gs, ids=[0, 2, 4], save_as=None):
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

    # base lattice
    diagnostics_lattice = lattice.copy()
    # params:
    params = torch.meshgrid(ks, vs, gs, indexing="ij")
    params = torch.stack(params, dim=-1).reshape((-1, 3)).unsqueeze(-1)
    diagnostics_lattice.elements[ids[0]].K1.data = params[:, 0].unsqueeze(-1)
    diagnostics_lattice.elements[ids[1]].VOLTAGE.data = params[:, 1].unsqueeze(-1)
    
    # change the dipole attributes + drift attribute
    G = params[:, 2].unsqueeze(-1)
    l_bend = 0.3018
    theta = torch.arcsin(l_bend * G) # AWA parameters
    l_arc = theta / G
    diagnostics_lattice.elements[ids[2]].G.data = G
    diagnostics_lattice.elements[ids[2]].L.data = l_arc
    diagnostics_lattice.elements[ids[2]].E2.data = theta
    
    diagnostics_lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)


    # track through lattice
    output_beam = diagnostics_lattice(beam)

    # histograms at screen
    images = screen(output_beam)

    # create image dataset
    dset = ImageDataset3D(params, images)

    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset saved as '{save_as}'")

    return dset


def run_awa_t_scan(beam, lattice, screen, ks, vs, gs, ids=[0, 2, 4], save_as=None):
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

    # base lattice
    diagnostics_lattice = lattice.copy()
    # params:
    # params = torch.meshgrid(ks, vs, gs, indexing='ij')
    # params = torch.stack(params, dim=-1).reshape((-1,3)).unsqueeze(-1)
    # allowed = torch.tensor([0, 4, 8, 9, 10, 11, 12, 16])
    n_ks = len(ks)
    params = torch.zeros((n_ks + 3, 3, 1))
    for i in range(n_ks):
        params[i, 0, 0] = ks[i]
        params[i, 1, 0] = vs[0]
        params[i, 2, 0] = gs[0]

    params[n_ks, 0, 0] = torch.tensor(0.0)
    params[n_ks, 1, 0] = vs[0]
    params[n_ks, 2, 0] = gs[1]

    params[n_ks + 1, 0, 0] = torch.tensor(0.0)
    params[n_ks + 1, 1, 0] = vs[1]
    params[n_ks + 1, 2, 0] = gs[0]

    params[n_ks + 2, 0, 0] = torch.tensor(0.0)
    params[n_ks + 2, 1, 0] = vs[1]
    params[n_ks + 2, 2, 0] = gs[1]

    print(params.shape)
    print(params[:, :, 0])
    
    diagnostics_lattice.elements[ids[0]].K1.data =  params[:, 0].unsqueeze(-1)
    diagnostics_lattice.elements[ids[1]].VOLTAGE.data =  params[:, 1].unsqueeze(-1)
    
    G = params[:, 2].unsqueeze(-1)
    l_bend = 0.3018
    theta = torch.arcsin(l_bend * G) # AWA parameters
    l_arc = theta / G
    diagnostics_lattice.elements[ids[2]].G.data = G
    diagnostics_lattice.elements[ids[2]].L.data = l_arc
    diagnostics_lattice.elements[ids[2]].E2.data = theta
    
    diagnostics_lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

    # track through lattice
    output_beam = diagnostics_lattice(beam)

    # histograms at screen
    images = screen(output_beam)

    # create image dataset
    dset = ImageDataset3D(params, images)

    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset saved as '{save_as}'")

    return dset


#### TEST ##################################################################################
def run_3d_scan_2screens(
    beam,
    lattice0,
    lattice1,
    screen0,
    screen1,
    #ks,
    #vs,
    #gs,
    params,
    n_imgs_per_param=1,
    ids=[0, 2, 4],
    save_as=None,
):
    """
    Runs 3D virtual scan (quad + TDC + dipole) and returns
    image dataset from the screen downstream.

    Parameters
    ----------
    beam : bmadx.Beam
        ground truth beam
    lattice: bmadx TorchLattice
        6D diagnostics lattice with quadrupole, TDC and dipole
    screen0: ImageDiagnostic
        Screen corresponding to dipole off
    screen1: ImageDiagnostic
        Screen corresponding to dipole on
    ks: torch.Tensor
        quadrupole strengths.
    vs: torch.Tensor
        TDC voltages.
    gs: torch.Tensor
        Dipole angles.
    n_imgs_per_param: int
        Number of images per parameter configuration.
    ids: list of ints
        Indices of the elements to be scanned: [quad_id, tdc_id, dipole_id]
    save_as : str
        filename to store output dataset. Default: None.

    Returns
    -------
    dset: ImageDataset
        scan dataset.images should be a 6D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of images per parameter configuration,
        screen width in pixels,
        screen height in pixels]
        train_dset.params should be a 4D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of scanning elements (3: quad, tdc, dipole) ]
    """

    # base lattices
    #params = torch.meshgrid(ks, vs, gs, indexing="ij")
    #params = torch.stack(params, dim=-1)
    params_dipole_off = params[:, :, 0].unsqueeze(-1)
    diagnostics_lattice0 = lattice0.copy()
    diagnostics_lattice0.elements[ids[0]].K1.data = params_dipole_off[:, :, 0]
    diagnostics_lattice0.elements[ids[1]].VOLTAGE.data = params_dipole_off[:, :, 1]
    # change the dipole attributes + drift attribute
    G = params_dipole_off[:, :, 2]
    l_bend = 0.3018
    theta = torch.arcsin(l_bend * G) # AWA parameters
    l_arc = theta / G
    diagnostics_lattice0.elements[ids[2]].G.data = G
    diagnostics_lattice0.elements[ids[2]].L.data = l_arc
    diagnostics_lattice0.elements[ids[2]].E2.data = theta
    diagnostics_lattice0.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

    params_dipole_on = params[:, :, 1].unsqueeze(-1)
    diagnostics_lattice1 = lattice1.copy()
    diagnostics_lattice1.elements[ids[0]].K1.data = params_dipole_on[:, :, 0]
    diagnostics_lattice1.elements[ids[1]].VOLTAGE.data = params_dipole_on[:, :, 1]
    # change the dipole attributes + drift attribute
    G = params_dipole_on[:, :, 2]
    l_bend = 0.3018
    theta = torch.arcsin(l_bend * G) # AWA parameters
    l_arc = theta / G
    diagnostics_lattice1.elements[ids[2]].G.data = G
    diagnostics_lattice1.elements[ids[2]].L.data = l_arc
    diagnostics_lattice1.elements[ids[2]].E2.data = theta
    diagnostics_lattice1.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

    # track through lattice for dipole off(0) and dipole on (1)
    output_beam0 = diagnostics_lattice0(beam)
    output_beam1 = diagnostics_lattice1(beam)

    # histograms at screens for dipole off(0) and dipole on (1)
    images_dipole_off = screen0(output_beam0).squeeze()
    images_dipole_on = screen1(output_beam1).squeeze()

    # stack on dipole dimension:
    images_stack = torch.stack((images_dipole_off, images_dipole_on), dim=2)

    # create images copies simulating multi-shot per parameter config:
    copied_images = torch.stack([images_stack] * n_imgs_per_param, dim=-3)

    # create image dataset
    dset = ImageDataset3D(params, copied_images)

    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset0 saved as '{save_as}'")

    return dset

