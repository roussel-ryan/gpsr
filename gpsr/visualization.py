def compare_beams(
    beam_1,
    beam_2,
    dimensions: tuple[str, ...] = ("x", "px", "y", "py", "tau", "p"),
    bins: int = 50,
    bin_ranges=None,
    contour_kws: dict = None,
):
    contour_kwargs = {
        "cmap": "Greys",
        "levels": [0.1, 0.5, 0.9],
        "zorder": 10,
    }
    contour_kwargs.update(contour_kws or {})

    fig, ax = beam_1.plot_distribution(
        dimensions=dimensions,
        bins=bins,
        bin_ranges=bin_ranges,
        plot_2d_kws={"pcolormesh_kws": {"cmap": "viridis"}},
    )
    plot_2d_kws = {"style": "contour", "contour_kws": contour_kwargs}
    beam_2.plot_distribution(
        dimensions=dimensions,
        bins=bins,
        bin_ranges=bin_ranges,
        axs=ax,
        plot_2d_kws=plot_2d_kws,
    )

    return fig, ax
