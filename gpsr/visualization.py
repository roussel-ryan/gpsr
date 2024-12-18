def compare_beams(
    beam_1,
    beam_2,
    coords: tuple[str, ...] = ("x", "px", "y", "py", "tau", "p"),
    bins: int = 50,
    custom_lims=None,
    levels=None,
    **kwargs,
):
    if levels is None:
        levels = [0.1, 0.5, 0.9]

    fig, ax = beam_1.plot_distribution(
        coords=coords, bins=bins, custom_lims=custom_lims
    )
    beam_2.plot_distribution(
        coords=coords,
        bins=bins,
        custom_lims=custom_lims,
        axes=ax,
        contour=True,
        zorder=10,
        cmap="Greys",
        levels=levels,
        **kwargs,
    )

    return fig, ax
