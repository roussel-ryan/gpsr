"""Screen-image plotting for GPSRLUME, driven by plain dicts and tensors.

``plot_images`` draws one grid of screen images, optionally overlaid with
peak-normalized contours -- of the same images (``contours=True``) or of a second
set passed as ``overlay_images``, for a measured-vs-predicted comparison.
``plot_ensemble_images`` draws the mean of an ensemble, optionally with
lower/mean/upper confidence-band contours. The two ``plot_multi_source_*``
wrappers fan those over a sources spec, one figure per source.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

from torch import Tensor

if TYPE_CHECKING:
    from tensordict import TensorDict

from gpsr.lume.ensemble import UncertaintyType, compute_mean_and_bounds

DEFAULT_IMAGE_CMAP = "Greys"
DEFAULT_CONTOUR_CMAP = "plasma"
DEFAULT_CONTOUR_LEVELS = (0.1, 0.5, 0.9)
DEFAULT_CONTOUR_SMOOTHING = 1.0


# ---------------------------------------------------------------------------
# Screen-image helpers
# ---------------------------------------------------------------------------
def _centered_edges(n: int, pixel_size: float) -> np.ndarray:
    """Return n+1 pixel edge positions of width ``pixel_size`` centered on 0."""
    return (np.arange(n + 1) - n / 2) * pixel_size


def _screen_extent(metadata: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_edges, y_edges) in mm for a screen image, centered on (0, 0).

    ``pixel_size`` (in metres) is read from ``metadata`` and converted to mm. A
    scalar pixel size is treated as square.
    """
    pixel_size = (
        np.asarray(metadata["pixel_size"], dtype=float).reshape(-1) * 1e3
    )  # m -> mm
    dx, dy = (
        (pixel_size[0], pixel_size[1])
        if pixel_size.size == 2
        else (pixel_size[0], pixel_size[0])
    )
    return _centered_edges(nx, dx), _centered_edges(ny, dy)


def _edges_to_centers(edges: np.ndarray) -> np.ndarray:
    """Return the n bin centers for n+1 monotonically spaced ``edges``."""
    return 0.5 * (edges[:-1] + edges[1:])


def _norm_to_peak(image: np.ndarray) -> np.ndarray:
    """Normalize an image to its own peak; all-zero images are left as zeros."""
    peak = image.max()
    return image / peak if peak > 0 else image


def _white_masked_cmap(cmap) -> mpl.colors.Colormap:
    """Return a copy of ``cmap`` (a name or an instance) with white masked pixels.

    Copied before mutating, so neither matplotlib's registry nor a caller's own
    colormap instance is left with a white "bad" color.
    """
    resolved = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    resolved = copy.copy(resolved)
    resolved.set_bad("white")
    return resolved


def _draw_screen_image(
    ax,
    image: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    vmax: float | None = None,
    pcolormesh_kwargs: dict | None = None,
) -> None:
    """Draw a single screen image on ``ax`` in physical (mm) units.

    ``image`` is in ``[x,y]`` order, so it is transposed for pcolormesh's
    ``[y,x]``. Axis labels are set per figure by ``_label_edge_axes``, not here.

    ``vmax`` is the grid-wide default; an explicit ``vmax`` in ``pcolormesh_kwargs``
    wins, including ``None`` for per-image autoscaling.

    ``pcolormesh_kwargs["white_background"]`` is popped rather than forwarded: it
    masks zero pixels, which also means the cmap must be resolved *after* the merge
    so one supplied in the same dict gets the white "bad" color.
    """
    kwargs = {
        "cmap": DEFAULT_IMAGE_CMAP,
        "vmin": 0,
        "vmax": vmax,
        "white_background": False,
    } | (pcolormesh_kwargs or {})
    if kwargs.pop("white_background"):
        image = np.ma.masked_where(image == 0, image)
        kwargs["cmap"] = _white_masked_cmap(kwargs["cmap"])
    ax.pcolormesh(x_edges, y_edges, image.T, **kwargs)
    ax.set_aspect("equal")


def _label_edge_axes(axes: np.ndarray) -> None:
    """Label mm axes on the figure edge only: x on the bottom row, y on the left col.

    ``axes`` is the 2D array from ``plt.subplots(..., squeeze=False)``.
    """
    for ax in axes[-1, :]:
        ax.set_xlabel("x (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (mm)")


def _resolve_single_key(mapping, name: str, key: str | None) -> str:
    """Resolve a key in ``mapping``, defaulting to the sole key when ``key`` is None."""
    keys = list(mapping.keys())
    if key is None:
        if len(keys) != 1:
            raise ValueError(
                f"A key must be given when {name} has more than one entry; "
                f"available keys: {keys}"
            )
        return keys[0]
    if key not in keys:
        raise KeyError(f"Unknown {name} key {key!r}; available keys: {keys}")
    return key


def _resolve_screen_key(
    mapping, name: str, key: str | None, observations_metadata: dict
) -> tuple[str, dict] | None:
    """Resolve an observation key + metadata, skipping non-screen observations.

    Returns ``(observation_key, metadata)``, or ``None`` after printing a skip
    message when the observation is not a screen.
    """
    observation_key = _resolve_single_key(mapping, name, key)
    metadata = observations_metadata[observation_key]
    if metadata.get("type") != "screen":
        print(
            f"Skipping {observation_key!r}: not a screen (type={metadata.get('type')!r})."
        )
        return None
    return observation_key, metadata


def _images_to_numpy(images, observation_key: str) -> np.ndarray:
    """Extract one observation's image stack as a detached numpy array."""
    return images[observation_key].detach().cpu().numpy()


def _resolve_label_values(
    beamline_settings, labels: str | list[str] | None
) -> dict[str, np.ndarray]:
    """Return ``{key: per-sample values}`` (detached numpy) for the column titles.

    ``labels`` may be a single key, a list of keys, or ``None`` for *all* settings
    keys. Empty when there is nothing to title with.
    """
    if beamline_settings is None:
        return {}
    if labels is None:
        keys = list(beamline_settings.keys())
    else:
        keys = [labels] if isinstance(labels, str) else list(labels)
    return {key: beamline_settings[key].detach().cpu().numpy() for key in keys}


def _screen_axes(
    image_stack: np.ndarray, metadata: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the pcolormesh mesh (edges + centers) for a screen image stack.

    ``image_stack``'s trailing two dims are ``(W, H)`` -- anything stacked in
    front (samples, draws) is ignored.
    """
    nx, ny = image_stack.shape[-2:]
    x_edges, y_edges = _screen_extent(metadata, nx, ny)
    return x_edges, y_edges, _edges_to_centers(x_edges), _edges_to_centers(y_edges)


def _load_overlay_images(
    overlay, observation_key: str, n_reference: int, reference_name: str
) -> np.ndarray | None:
    """Load ``overlay`` for ``observation_key`` and check its scan-step count.

    ``None`` for a ``None`` overlay; raises ``ValueError`` when the leading dim
    does not match ``n_reference``. ``reference_name`` names the compared-against
    set in that error.
    """
    if overlay is None:
        return None
    arr = _images_to_numpy(overlay, observation_key)
    if arr.shape[0] != n_reference:
        raise ValueError(
            f"overlay_images has a different sample count than the "
            f"{reference_name} for {observation_key!r}: {arr.shape[0]} vs "
            f"{n_reference}. They must be row-aligned (same scan steps in the "
            f"same order)."
        )
    return arr


def _format_title(label_values: dict[str, np.ndarray], col: int) -> str:
    """Multi-line ``'key = value'`` title for scan step ``col`` (one line per key).

    Empty string when there are no keys, so callers can skip drawing a title.
    """
    return "\n".join(
        f"{key} = {float(values[col]):.2f}" for key, values in label_values.items()
    )


def _draw_peak_normalized_contour(
    ax,
    image: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    linestyle: str,
    alpha: float = 1.0,
    contour_kwargs: dict | None = None,
) -> None:
    """Smooth, peak-normalize and contour ``image`` at fractional ``levels``.

    ``linestyle`` encodes the role: dashed = reference / band, solid = overlay.
    ``contour_kwargs["smoothing"]`` is popped rather than forwarded, since it
    transforms the data and ``Axes.contour`` raises on an unknown kwarg.
    """
    kwargs = {
        "cmap": DEFAULT_CONTOUR_CMAP,
        "levels": list(DEFAULT_CONTOUR_LEVELS),
        "vmin": 0,
        "vmax": 1.0,
        "linestyles": linestyle,
        "alpha": alpha,
        "smoothing": DEFAULT_CONTOUR_SMOOTHING,
    } | (contour_kwargs or {})

    smoothing = kwargs.pop("smoothing")
    if smoothing is not None:
        image = gaussian_filter(image, smoothing)
    ax.contour(x_centers, y_centers, _norm_to_peak(image.T), **kwargs)


def _plot_screen_grid(
    fill_images,
    x_edges,
    y_edges,
    observation_key,
    label_values,
    n_cols,
    vmax,
    x_range,
    y_range,
    draw_overlays=None,
    pcolormesh_kwargs=None,
    title: bool = True,
    title_prefix: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Common layout for a per-scan-step grid of filled screen images.

    Draws one filled ``pcolormesh`` cell per sample, plus whatever the
    ``draw_overlays(ax, col)`` callback adds per cell. Pass ``label_values={}`` to
    suppress per-column titles, ``title=False`` for no suptitle.
    """
    n_samples = len(fill_images)
    if n_cols is None:
        n_cols = n_samples

    n_rows = int(np.ceil(n_samples / n_cols))
    if figsize is None:
        figsize = (2.5 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex="all", sharey="all", squeeze=False
    )
    flat_axes = axes.flatten()

    for col, (ax, image) in enumerate(zip(flat_axes, fill_images)):
        _draw_screen_image(
            ax,
            image,
            x_edges,
            y_edges,
            vmax=vmax,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )
        if draw_overlays is not None:
            draw_overlays(ax, col)
        column_title = _format_title(label_values, col)
        if column_title:
            ax.set_title(column_title, fontsize=mpl.rcParams["axes.labelsize"])

    for ax in flat_axes[n_samples:]:
        ax.axis("off")

    if x_range is not None:
        axes[0, 0].set_xlim(x_range)
    if y_range is not None:
        axes[0, 0].set_ylim(y_range)

    _label_edge_axes(axes)
    if title:
        fig.suptitle(
            f"{title_prefix} - {observation_key}" if title_prefix else observation_key
        )
    fig.tight_layout()
    return fig


def _fan_over_sources(
    plot_fn,
    images: dict[str, dict],
    sources_spec: dict[str, dict],
    overlay_images: dict[str, dict] | None,
    images_field: str,
    **plot_kwargs,
) -> dict[str, plt.Figure | None]:
    """Fan a per-source images dict out to a single-source plotting function.

    ``images_field`` names the fanned-out dict in the missing-source errors.
    """
    figures: dict[str, plt.Figure | None] = {}
    for source_name, source_images in images.items():
        if source_name not in sources_spec:
            raise KeyError(
                f"sources_spec is missing source {source_name!r} present in "
                f"{images_field}; available: {sorted(sources_spec.keys())}."
            )
        source = sources_spec[source_name]
        if overlay_images is None:
            overlay = None
        elif source_name not in overlay_images:
            raise KeyError(
                f"overlay_images is missing source {source_name!r} present "
                f"in {images_field}; available: {sorted(overlay_images.keys())}."
            )
        else:
            overlay = overlay_images[source_name]
        figures[source_name] = plot_fn(
            source["beamline_settings"],
            source_images,
            source["observations_metadata"],
            overlay_images=overlay,
            # Source name as the default prefix; a caller's own title_prefix wins.
            **{"title_prefix": source_name, **plot_kwargs},
        )
    return figures


# ---------------------------------------------------------------------------
# Screen-image public API
# ---------------------------------------------------------------------------
def plot_images(
    beamline_settings: dict[str, Tensor] | TensorDict | None,
    images: dict[str, Tensor] | TensorDict,
    observations_metadata: dict[str, dict],
    overlay_images: dict[str, Tensor] | TensorDict | None = None,
    observation_key: str | None = None,
    labels: str | list[str] | None = None,
    n_cols: int | None = None,
    contours: bool = False,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    title: bool = True,
    title_prefix: str | None = None,
    figsize: tuple[float, float] | None = None,
    pcolormesh_kwargs: dict | None = None,
    contour_kwargs: dict | None = None,
    overlay_contour_kwargs: dict | None = None,
):
    """Plot the images of a 'screen' observation, optionally with contour overlays.

    The three positional arguments are the triple ``GPSRLUMEDataset.to_dict``
    holds, so a dataset splats straight in and a measured-vs-predicted comparison
    reads ``plot_images(**dataset.to_dict(), overlay_images=preds)``. Measured and
    predicted images fill the same ``images`` slot.

    Three drawing modes, all over a ``pcolormesh`` fill of the ``images`` set:

    - Default: fill only.
    - ``contours=True``: each image also drawn as peak-normalized dashed contours.
    - ``overlay_images`` given: one row per scan step, with the overlay as contour
      lines on the same axes. Each image is normalized to its own peak, so the fill
      autoscales per image unless ``pcolormesh_kwargs["vmax"]`` says otherwise.

    Linestyle encodes the source dict: dashed = the first-arg ``images`` set
    (reference / reconstruction), solid = the ``overlay_images`` ground-truth
    set. Same convention used by ``plot_ensemble_images``.

    Axes are in physical units (mm) using the observation's ``pixel_size``
    metadata and centered on (0, 0). Non-screen observations are skipped.

    Parameters
    ----------
    beamline_settings : dict[str, Tensor] | TensorDict | None
        Scan parameters keyed by PV name; each tensor's leading dim is n_samples.
        Used only for per-image/column titles (see ``labels``). Pass ``None`` to
        draw no titles.
    images : dict[str, Tensor] | TensorDict
        Images keyed by PV name (the reference set: the single grid, or the
        filled images in the overlay comparison). Leading dim is n_samples.
    observations_metadata : dict[str, dict]
        Per-observation metadata; the selected key must have ``type == "screen"``
        and a ``pixel_size``.
    overlay_images : dict[str, Tensor] | TensorDict | None
        Optional second image set (e.g. predictions, or measured images to
        overlay on predictions) under the *same* ``observation_key`` and with
        the same per-sample count. If given, it is overlaid as contour lines on
        the ``images`` fills.
    observation_key : str | None
        Which observation (PV) to plot. If None and there is exactly one
        observation, that key is used.
    labels : str | list[str] | None
        Beamline-settings key(s) whose per-sample values title each image (single
        grid) or each column (overlay comparison), one ``"key = value"`` line per
        key. If None (default), *all* settings keys are shown; pass ``[]`` for no
        per-panel titles. Titles are drawn only when ``beamline_settings`` is given.
    n_cols : int | None
        Number of columns in the single-grid layout. If None (default), all images
        are placed in a single row. Ignored in comparison mode (columns are fixed
        to one per scan step).
    contours : bool
        Overlay the ``images`` set as peak-normalized dashed contours on top of the
        fill. Defaults to False.
    x_range : tuple[float, float] | None
        Horizontal axis limits in mm, e.g. ``(-5, 5)``. If None (default), the
        full image extent is shown.
    y_range : tuple[float, float] | None
        Vertical axis limits in mm. If None (default), the full image extent is
        shown.
    title : bool
        If True (default), draw the ``observation_key`` as the figure suptitle;
        if False, no suptitle is drawn.
    title_prefix : str | None
        Prepended to the suptitle as ``"{title_prefix} - {observation_key}"``.
        ``plot_multi_source_images`` passes the source name.
    figsize : tuple[float, float] | None
        Figure size ``(width, height)`` in inches. If None (default), auto-sized
        as ``(2.5 * n_cols, 4 * n_rows)``.
    pcolormesh_kwargs : dict | None
        Extra kwargs for ``Axes.pcolormesh``, merged over the fill defaults
        (``cmap="Greys"``, ``vmin=0``, ``vmax`` the max over all images), so user
        keys win. Pass ``vmax=None`` to scale each image to its own peak instead of
        a common scale; with ``overlay_images`` that is already the default. The
        extra key ``white_background`` (default False) masks zero pixels white
        instead of using the colormap's low end; a ``cmap`` in the same dict gets
        that white masked colour too.
    contour_kwargs : dict | None
        Extra kwargs for ``Axes.contour`` on the ``images`` set, merged over the
        defaults (``cmap="plasma"``, ``levels=(0.1, 0.5, 0.9)``, dashed). The extra
        key ``smoothing`` (default 1.0, ``None`` to disable) Gaussian-smooths the
        contour source first, to tame noisy contours.
    overlay_contour_kwargs : dict | None
        The same, for the ``overlay_images`` contours (solid), so the two sets can
        be smoothed independently.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or None if the observation is not a screen.
    """
    resolved = _resolve_screen_key(
        images, "images", observation_key, observations_metadata
    )
    if resolved is None:
        return None
    observation_key, metadata = resolved

    images = _images_to_numpy(images, observation_key)
    x_edges, y_edges, x_centers, y_centers = _screen_axes(images, metadata)
    label_values = _resolve_label_values(beamline_settings, labels)
    overlay = _load_overlay_images(
        overlay_images, observation_key, images.shape[0], "images"
    )

    # Contours are each self-normalized, so per-image normalize the fill too.
    vmax = None if overlay is not None else np.max(images)

    def draw_overlays(ax, col):
        reference_kws = contour_kwargs or {}
        overlay_kws = overlay_contour_kwargs or {}
        if overlay is None:
            if contours:
                _draw_peak_normalized_contour(
                    ax,
                    images[col],
                    x_centers,
                    y_centers,
                    linestyle="dashed",
                    contour_kwargs=reference_kws,
                )
            return
        for source, linestyle, role_kws in (
            (images[col], "dashed", reference_kws),
            (overlay[col], "solid", overlay_kws),
        ):
            _draw_peak_normalized_contour(
                ax,
                source,
                x_centers,
                y_centers,
                linestyle=linestyle,
                contour_kwargs=role_kws,
            )

    return _plot_screen_grid(
        images,
        x_edges,
        y_edges,
        observation_key=observation_key,
        label_values=label_values,
        n_cols=n_cols,
        vmax=vmax,
        x_range=x_range,
        y_range=y_range,
        draw_overlays=draw_overlays,
        pcolormesh_kwargs=pcolormesh_kwargs,
        title=title,
        title_prefix=title_prefix,
        figsize=figsize,
    )


def plot_ensemble_images(
    beamline_settings: dict | None,
    ensemble_images: dict,
    observations_metadata: dict,
    overlay_images: dict | None = None,
    observation_key: str | None = None,
    labels: str | list[str] | None = None,
    n_cols: int | None = None,
    band: bool = True,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    title: bool = True,
    title_prefix: str | None = None,
    figsize: tuple[float, float] | None = None,
    pcolormesh_kwargs: dict | None = None,
    contour_kwargs: dict | None = None,
    overlay_contour_kwargs: dict | None = None,
) -> plt.Figure | None:
    """Plot ensemble predicted screen images: mean fill + confidence-band contours.

    Takes an already-predicted ensemble rather than a model and beam:

        preds = predict_ensemble_images(model, settings, metadata, beam=ensemble_beam)
        plot_ensemble_images(settings, preds, metadata,
                             overlay_images=dataset.data["observations"])

    Each per-PV stack ``(n_draws, n_samples, W, H)`` is reduced over the draw dim
    with ``compute_mean_and_bounds``, then one column per scan step is drawn: the
    mean image filled, and -- when ``band=True`` (default) -- the lower, mean and
    upper histograms each overlaid as a dashed contour, self-normalized to its own
    peak. ``band=False`` draws only the mean fill. ``overlay_images`` is drawn as
    solid contours either way.

    Axes are in physical mm (from ``pixel_size``).

    Parameters
    ----------
    beamline_settings : dict[str, Tensor] | None
        Scan parameters keyed by PV name; used only for per-column titles (see
        ``labels``). Pass ``None`` to draw no titles.
    ensemble_images : dict[str, Tensor]
        Ensemble predicted images keyed by observation PV, each of shape
        ``(n_draws, n_samples, W, H)`` -- the output of
        ``predict_ensemble_images``.
    observations_metadata : dict[str, dict]
        Per-observation metadata; the selected key must have ``type == "screen"``
        and a ``pixel_size``.
    overlay_images : dict[str, Tensor] | None
        Optional single-sample-per-step images (e.g. measured) under the same
        ``observation_key`` and per-sample (scan-step) count, overlaid as solid
        contours.
    observation_key : str | None
        Which observation (PV) to plot. If None and there is exactly one entry in
        ``ensemble_images``, that key is used.
    labels : str | list[str] | None
        Beamline-settings key(s) whose per-sample values title each column, one
        ``"key = value"`` line per key. If None (default), *all* settings keys are
        shown; pass ``[]`` for no per-panel titles.
    n_cols : int | None
        Number of columns in the image grid. ``None`` (default) places all
        samples in a single row.
    band : bool
        If True (default), overlay the ensemble band (lower / mean / upper) as
        dashed peak-normalized contours on the mean fill; if False, draw only the
        fill. ``overlay_images`` is drawn as solid contours either way.
    uncertainty_type : "percentile" | "std_error"
        Passed to ``compute_mean_and_bounds``.
    confidence_level : float
        Confidence level for the band.
    x_range : tuple[float, float] | None
        Horizontal axis limits in mm. If None (default), the full image extent
        is shown.
    y_range : tuple[float, float] | None
        Vertical axis limits in mm. If None (default), the full image extent
        is shown.
    title : bool
        If True (default), draw the ``observation_key`` as the figure suptitle;
        if False, no suptitle is drawn.
    title_prefix : str | None
        Prepended to the suptitle as ``"{title_prefix} - {observation_key}"``.
        ``plot_multi_source_ensemble_images`` passes the source name.
    figsize : tuple[float, float] | None
        Figure size ``(width, height)`` in inches. If None (default), auto-sized
        as ``(2.5 * n_cols, 4 * n_rows)``.
    pcolormesh_kwargs : dict | None
        Extra kwargs for ``Axes.pcolormesh``, merged over the mean-image fill
        defaults (``cmap="Greys"``, ``vmin=0``, ``vmax`` the max over all mean
        images), so user keys win. Pass ``vmax=None`` to scale each mean image to
        its own peak instead of a common scale; the band and overlay contours are
        self-normalized either way. The extra key ``white_background`` (default
        False) masks zero pixels white instead of using the colormap's low end; a
        ``cmap`` in the same dict gets that white masked colour too.
    contour_kwargs : dict | None
        Extra kwargs for ``Axes.contour`` on the ensemble band, merged over the
        defaults (``cmap="plasma"``, ``levels=(0.1, 0.5, 0.9)``, dashed). The extra
        key ``smoothing`` (default 1.0, ``None`` to disable) Gaussian-smooths the
        contour source first. Ignored when ``band=False``.
    overlay_contour_kwargs : dict | None
        The same, for the ``overlay_images`` contours (solid), so the band and the
        overlay can be smoothed independently.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or None if the observation is not a screen.
    """
    resolved = _resolve_screen_key(
        ensemble_images,
        "ensemble_images",
        observation_key,
        observations_metadata,
    )
    if resolved is None:
        return None
    observation_key, metadata = resolved

    # (n_draws, n_samples, W, H) -> mean / bounds over draws: (n_samples, W, H)
    mean, lower, upper = compute_mean_and_bounds(
        torch.as_tensor(ensemble_images[observation_key]),
        uncertainty_type=uncertainty_type,
        confidence_level=confidence_level,
    )
    mean = mean.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()
    upper = upper.detach().cpu().numpy()

    x_edges, y_edges, x_centers, y_centers = _screen_axes(mean, metadata)
    label_values = _resolve_label_values(beamline_settings, labels)
    overlay = _load_overlay_images(
        overlay_images, observation_key, mean.shape[0], "ensemble_images"
    )
    vmax = np.max(mean)

    def draw_overlays(ax, col):
        band_kws = contour_kwargs or {}
        overlay_kws = overlay_contour_kwargs or {}
        if band:
            for source in (lower[col], mean[col], upper[col]):
                _draw_peak_normalized_contour(
                    ax,
                    source,
                    x_centers,
                    y_centers,
                    linestyle="dashed",
                    alpha=0.75,
                    contour_kwargs=band_kws,
                )
        if overlay is not None:
            _draw_peak_normalized_contour(
                ax,
                overlay[col],
                x_centers,
                y_centers,
                linestyle="solid",
                alpha=1.0,
                contour_kwargs=overlay_kws,
            )

    return _plot_screen_grid(
        mean,
        x_edges,
        y_edges,
        observation_key=observation_key,
        label_values=label_values,
        n_cols=n_cols,
        vmax=vmax,
        x_range=x_range,
        y_range=y_range,
        draw_overlays=draw_overlays,
        pcolormesh_kwargs=pcolormesh_kwargs,
        title=title,
        title_prefix=title_prefix,
        figsize=figsize,
    )


def plot_multi_source_images(
    images: dict[str, dict],
    sources_spec: dict[str, dict],
    overlay_images: dict[str, dict] | None = None,
    **kwargs,
) -> dict[str, plt.Figure | None]:
    """Plot single-beam predicted screen images for every source, one figure per source.

    Draws each source with ``plot_images``:

        spec = dm.to_sources_spec()
        preds = predict_multi_source_images(model, spec, beam=beam)
        plot_multi_source_images(preds, spec)
        plot_multi_source_images(preds, spec, overlay_images=dm.to_observations())

    Each suptitle is ``"<source> - <observation key>"``, since sources typically
    observe the same screen.

    Parameters
    ----------
    images : dict[str, dict[str, Tensor]]
        Per source, images keyed by observation PV, each of shape
        ``(n_samples, W, H)``.
    sources_spec : dict[str, dict]
        Per-source spec supplying each source's ``beamline_settings`` and
        ``observations_metadata``. Must cover every source in ``images``.
    overlay_images : dict[str, dict[str, Tensor]] | None
        Optional overlay images per source, drawn as solid contours. If given,
        must cover every source in ``images``.
    **kwargs
        Forwarded to ``plot_images`` for every source. ``title_prefix`` defaults
        to each source's own name.

    Returns
    -------
    dict[str, matplotlib.figure.Figure | None]
        Per source: the created figure, or None if that source's observation is
        not a screen.
    """
    return _fan_over_sources(
        plot_images,
        images,
        sources_spec,
        overlay_images,
        "images",
        **kwargs,
    )


def plot_multi_source_ensemble_images(
    ensemble_images: dict[str, dict],
    sources_spec: dict[str, dict],
    overlay_images: dict[str, dict] | None = None,
    **kwargs,
) -> dict[str, plt.Figure | None]:
    """Plot ensemble predicted screen images for every source, one figure per source.

    Draws each source with ``plot_ensemble_images`` (mean fill + confidence-band
    contours):

        spec = dm.to_sources_spec()
        preds = predict_multi_source_ensemble_images(model, spec, beam)
        plot_multi_source_ensemble_images(preds, spec)
        plot_multi_source_ensemble_images(
            preds, spec, overlay_images=dm.to_observations()
        )

    Each suptitle is ``"<source> - <observation key>"``, since sources typically
    observe the same screen.

    Parameters
    ----------
    ensemble_images : dict[str, dict[str, Tensor]]
        Per source, ensemble predicted images keyed by observation PV, each of
        shape ``(n_draws, n_samples, W, H)``.
    sources_spec : dict[str, dict]
        Per-source spec supplying each source's ``beamline_settings`` and
        ``observations_metadata``. Must cover every source in ``ensemble_images``.
    overlay_images : dict[str, dict[str, Tensor]] | None
        Optional overlay images per source, drawn as solid contours. If given,
        must cover every source in ``ensemble_images``.
    **kwargs
        Forwarded to ``plot_ensemble_images`` for every source. ``title_prefix``
        defaults to each source's own name.

    Returns
    -------
    dict[str, matplotlib.figure.Figure | None]
        Per source: the created figure, or None if that source's observation is
        not a screen.
    """
    return _fan_over_sources(
        plot_ensemble_images,
        ensemble_images,
        sources_spec,
        overlay_images,
        "ensemble_images",
        **kwargs,
    )
