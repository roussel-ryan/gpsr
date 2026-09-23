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


def _screen_cmap(name: str):
    """Return the colormap ``name`` with white for masked (zero) pixels."""
    cmap = copy.copy(mpl.colormaps[name])
    cmap.set_bad("white")
    return cmap


def _draw_screen_image(
    ax,
    image: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cmap,
    vmax: float | None = None,
    white_background: bool = False,
    alpha: float = 1.0,
    pcolormesh_kwargs: dict | None = None,
) -> None:
    """Draw a single screen image on ``ax`` in physical (mm) units.

    When ``white_background`` is True, zero pixels are masked (shown as the
    colormap's white "bad" color) instead of the colormap's low end. ``alpha``
    sets the fill opacity. Shares the pcolormesh / aspect styling used across all
    GPSRLUME image plots. Axis labels are set once per figure by
    ``_label_edge_axes``, not here.
    Note: image is in [x,y] ordering, so it is transposed for pcolormesh (which expects [y,x]).
    """
    if white_background:
        image = np.ma.masked_where(image == 0, image)
    defaults = {"cmap": cmap, "vmin": 0, "vmax": vmax, "alpha": alpha}
    ax.pcolormesh(x_edges, y_edges, image.T, **(defaults | (pcolormesh_kwargs or {})))
    ax.set_aspect("equal")


def _label_edge_axes(axes: np.ndarray) -> None:
    """Label mm axes on the figure edge only: x on the bottom row, y on the left col.

    ``axes`` is the 2D array from ``plt.subplots(..., squeeze=False)``. With shared
    axes this keeps columns tight (no repeated per-axis labels between them).
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

    Returns ``(observation_key, metadata)`` or ``None`` when the observation is
    not a screen (a skip message is printed in that case). Encapsulates the guard
    both public image plotters open with.
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


def _resolve_label_keys(beamline_settings, labels: str | list[str] | None) -> list[str]:
    """Resolve which beamline-settings keys title each image / column.

    ``labels`` may be a single key, a list of keys, or ``None`` (use *all*
    settings keys). Returns ``[]`` when there are no settings to title with.
    """
    if beamline_settings is None:
        return []
    if labels is None:
        return list(beamline_settings.keys())
    return [labels] if isinstance(labels, str) else list(labels)


def _label_values(beamline_settings, label_keys: list[str]) -> dict[str, np.ndarray]:
    """Return ``{key: per-sample values}`` (detached numpy) for ``label_keys``."""
    return {key: beamline_settings[key].detach().cpu().numpy() for key in label_keys}


def _resolve_label_values(
    beamline_settings, labels: str | list[str] | None
) -> dict[str, np.ndarray]:
    """One-shot ``{key: per-sample values}`` for column titles.

    Combines ``_resolve_label_keys`` and ``_label_values`` -- the
    two-step call both public image plotters otherwise repeat verbatim.
    """
    return _label_values(
        beamline_settings, _resolve_label_keys(beamline_settings, labels)
    )


def _screen_axes(
    image_stack: np.ndarray, metadata: dict, image_cmap: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, mpl.colors.Colormap]:
    """Return the pcolormesh mesh (edges + centers) + fill cmap for a screen stack.

    ``image_stack``'s trailing two dims are ``(W, H)`` -- anything stacked in
    front (samples, draws) is ignored. Bundles the four-line setup both public
    image plotters share.
    """
    nx, ny = image_stack.shape[-2:]
    x_edges, y_edges = _screen_extent(metadata, nx, ny)
    x_centers, y_centers = _edges_to_centers(x_edges), _edges_to_centers(y_edges)
    return x_edges, y_edges, x_centers, y_centers, _screen_cmap(image_cmap)


def _load_overlay_images(
    overlay, observation_key: str, n_reference: int, reference_name: str
) -> np.ndarray | None:
    """Load ``overlay`` for ``observation_key`` and check its scan-step count.

    Returns ``None`` if ``overlay`` is ``None`` (nothing to overlay). Raises
    ``ValueError`` when the loaded stack's leading dim does not match
    ``n_reference``. ``reference_name`` (``"images"`` / ``"ensemble_images"``)
    is used only in the error message.
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
    cmap: str,
    levels: list[float],
    linestyle: str,
    smoothing: float | None,
    alpha: float = 1.0,
    contour_kwargs: dict | None = None,
) -> None:
    """Peak-normalize ``image`` (optionally smoothing first) and draw as contours.

    Shared by both image plotters: the source is optionally Gaussian-smoothed,
    transposed to pcolormesh's ``[y,x]`` ordering, normalized to its own peak,
    then contoured at fractional ``levels`` using ``cmap`` mapped to ``[0, 1]``.
    """
    if smoothing is not None:
        image = gaussian_filter(image, smoothing)
    defaults = {
        "cmap": cmap,
        "levels": levels,
        "vmin": 0,
        "vmax": 1.0,
        "linestyles": linestyle,
        "alpha": alpha,
    }
    ax.contour(
        x_centers,
        y_centers,
        _norm_to_peak(image.T),
        **(defaults | (contour_kwargs or {})),
    )


def _plot_screen_grid(
    fill_images,
    x_edges,
    y_edges,
    cmap,
    observation_key,
    label_values,
    n_cols,
    vmax,
    white_background,
    image_alpha,
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
    ``draw_overlays(ax, col)`` callback adds per cell -- contours for
    ``plot_images``, the confidence band for ``plot_ensemble_images``. Pass
    ``label_values={}`` to suppress per-column titles, ``title=False`` for no
    suptitle.
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
            cmap,
            vmax=vmax,
            white_background=white_background,
            alpha=image_alpha,
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

    ``images_field`` is used only in the missing-source error message. Each
    figure's suptitle is prefixed with its source name, since ``observation_key``
    is usually the same screen across sources.
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
            # Merged (rather than setdefault'd into plot_kwargs) so the prefix is
            # recomputed per source and an explicit caller-supplied title_prefix
            # still wins without colliding as a duplicate keyword.
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
    normalize_each: bool = False,
    contours: bool = False,
    contour_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    contour_smoothing: float | None = 1.0,
    white_background: bool = False,
    image_cmap: str = "Greys",
    contour_cmap: str = "plasma",
    image_alpha: float = 1.0,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    title: bool = True,
    title_prefix: str | None = None,
    label_settings: bool = True,
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
      lines on the same axes. Each image is normalized to its own peak, so
      ``normalize_each`` is ignored.

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
        key. If None (default), *all* settings keys are shown. Pass a single key or
        a list to restrict which settings are shown; titles are drawn only when
        ``beamline_settings`` is given.
    n_cols : int | None
        Number of columns in the single-grid layout. If None (default), all images
        are placed in a single row. Ignored in comparison mode (columns are fixed
        to one per scan step).
    normalize_each : bool
        Single-grid only. If True, each image is scaled independently; if False, a
        common vmax is used across all images. Ignored by the overlay comparison,
        which always per-image normalizes.
    contours : bool
        Overlay the ``images`` set as peak-normalized dashed contours on top of the
        fill. Defaults to False.
    contour_levels : tuple[float, ...]
        Contour levels as fractions of each image's peak. Defaults to
        (0.1, 0.5, 0.9).
    contour_smoothing : float | None
        Gaussian-smooth by this sigma before contouring, to tame noisy contours.
        Defaults to 1.0.
    white_background : bool
        If True, masked (zero) pixels are drawn white instead of the colormap's low
        end. Defaults to False.
    image_cmap : str
        Colormap for the filled ``pcolormesh`` images. Defaults to ``"Greys"``.
    contour_cmap : str
        Colormap for the contour lines (both the ``contours=True`` overlay and
        the overlay comparison set). Defaults to ``"plasma"``.
    image_alpha : float
        Opacity of the filled ``pcolormesh`` images, in ``[0, 1]``. Defaults to
        1.0 (opaque).
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
    label_settings : bool
        If True (default), title each image / column with its beamline-settings
        values; if False, draw no per-panel titles.
    figsize : tuple[float, float] | None
        Figure size ``(width, height)`` in inches. If None (default), auto-sized
        as ``(2.5 * n_cols, 4 * n_rows)``.
    pcolormesh_kwargs : dict | None
        Extra kwargs for ``Axes.pcolormesh``, merged over the defaults built from
        ``image_cmap`` / ``image_alpha`` / ``vmax``, so user keys win.
    contour_kwargs : dict | None
        Extra kwargs for ``Axes.contour`` on the ``images`` set (dashed), merged
        over the defaults from ``contour_cmap`` / ``contour_levels``.
    overlay_contour_kwargs : dict | None
        The same, for the ``overlay_images`` contours (solid).

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
    x_edges, y_edges, x_centers, y_centers, cmap = _screen_axes(
        images, metadata, image_cmap
    )
    label_values = (
        _resolve_label_values(beamline_settings, labels) if label_settings else {}
    )
    overlay = _load_overlay_images(
        overlay_images, observation_key, images.shape[0], "images"
    )

    # Contours are each self-normalized, so per-image normalize the fill too.
    normalize_each = normalize_each or overlay is not None
    vmax = None if normalize_each else np.max(images)
    levels = list(contour_levels)

    def draw_overlays(ax, col):
        reference_kws = contour_kwargs or {}
        overlay_kws = overlay_contour_kwargs or {}
        if overlay is None:
            if contours:
                # No comparison: draw the reference (first-arg) as dashed
                # contours -- linestyle encodes the source dict (dashed =
                # first-arg / reconstruction, solid = overlay_images).
                _draw_peak_normalized_contour(
                    ax,
                    images[col],
                    x_centers,
                    y_centers,
                    cmap=contour_cmap,
                    levels=levels,
                    linestyle="dashed",
                    smoothing=contour_smoothing,
                    contour_kwargs=reference_kws,
                )
            return
        # Reference contours (dashed) and overlay-set contours (solid), each
        # peak-normalized. Mirrors ``gpsr.datasets.QuadScanDataset.plot_data``.
        for source, linestyle, role_kws in (
            (images[col], "dashed", reference_kws),
            (overlay[col], "solid", overlay_kws),
        ):
            _draw_peak_normalized_contour(
                ax,
                source,
                x_centers,
                y_centers,
                cmap=contour_cmap,
                levels=levels,
                linestyle=linestyle,
                smoothing=contour_smoothing,
                contour_kwargs=role_kws,
            )

    return _plot_screen_grid(
        images,
        x_edges,
        y_edges,
        cmap,
        observation_key=observation_key,
        label_values=label_values,
        n_cols=n_cols,
        vmax=vmax,
        white_background=white_background,
        image_alpha=image_alpha,
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
    normalize_each: bool = False,
    band: bool = True,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.8,
    contour_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    contour_smoothing: float | None = 1.0,
    white_background: bool = False,
    image_cmap: str = "Greys",
    contour_cmap: str = "plasma",
    image_alpha: float = 1.0,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    title: bool = True,
    title_prefix: str | None = None,
    label_settings: bool = True,
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
        shown. Pass a single key or a list to restrict which settings are shown.
    n_cols : int | None
        Number of columns in the image grid. ``None`` (default) places all
        samples in a single row.
    normalize_each : bool
        Governs the mean-image fill only (the band / overlay contours are always
        self-normalized). If True, each mean image is scaled independently; if
        False (default), a common vmax is used across all mean images.
    band : bool
        If True (default), overlay the ensemble band (lower / mean / upper) as
        dashed peak-normalized contours on the mean fill; if False, draw only the
        fill. ``overlay_images`` is drawn as solid contours either way.
    uncertainty_type : "percentile" | "std_error"
        Passed to ``compute_mean_and_bounds``.
    confidence_level : float
        Confidence level for the band.
    contour_levels : tuple[float, ...]
        Contour levels (fractions of each image's peak) for the band / overlay.
    contour_smoothing : float | None
        If given, contour sources are Gaussian-smoothed by this sigma first.
    white_background : bool
        If True, masked (zero) pixels of the mean image are drawn white.
    image_cmap : str
        Colormap for the filled mean image. Defaults to ``"Greys"``.
    contour_cmap : str
        Colormap for the contour lines. Defaults to ``"plasma"``.
    image_alpha : float
        Opacity of the filled mean image.
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
    label_settings : bool
        If True (default), title each column with its beamline-settings values; if
        False, draw no per-panel titles.
    figsize : tuple[float, float] | None
        Figure size ``(width, height)`` in inches. If None (default), auto-sized
        as ``(2.5 * n_cols, 4 * n_rows)``.
    pcolormesh_kwargs : dict | None
        Extra kwargs for ``Axes.pcolormesh``, merged over the defaults built from
        ``image_cmap`` / ``image_alpha`` / ``vmax``, so user keys win.
    contour_kwargs : dict | None
        Extra kwargs for ``Axes.contour`` on the ensemble band (dashed). Ignored
        when ``band=False``.
    overlay_contour_kwargs : dict | None
        The same, for the ``overlay_images`` contours (solid).

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

    x_edges, y_edges, x_centers, y_centers, cmap = _screen_axes(
        mean, metadata, image_cmap
    )
    label_values = (
        _resolve_label_values(beamline_settings, labels) if label_settings else {}
    )
    overlay = _load_overlay_images(
        overlay_images, observation_key, mean.shape[0], "ensemble_images"
    )
    vmax = None if normalize_each else np.max(mean)
    levels = list(contour_levels)

    def draw_overlays(ax, col):
        band_kws = contour_kwargs or {}
        overlay_kws = overlay_contour_kwargs or {}
        if band:
            # Ensemble band: lower / mean / upper each as a dashed contour over
            # the mean fill -- three dashed rings per level bracketing the mean,
            # mirroring the 3-contour style of plot_2d_distribution.
            for source in (lower[col], mean[col], upper[col]):
                _draw_peak_normalized_contour(
                    ax,
                    source,
                    x_centers,
                    y_centers,
                    cmap=contour_cmap,
                    levels=levels,
                    linestyle="dashed",
                    smoothing=contour_smoothing,
                    alpha=0.75,
                    contour_kwargs=band_kws,
                )
        # Overlay (if given) as solid ground-truth contours.
        if overlay is not None:
            _draw_peak_normalized_contour(
                ax,
                overlay[col],
                x_centers,
                y_centers,
                cmap=contour_cmap,
                levels=levels,
                linestyle="solid",
                smoothing=contour_smoothing,
                alpha=1.0,
                contour_kwargs=overlay_kws,
            )

    return _plot_screen_grid(
        mean,
        x_edges,
        y_edges,
        cmap,
        observation_key=observation_key,
        label_values=label_values,
        n_cols=n_cols,
        vmax=vmax,
        white_background=white_background,
        image_alpha=image_alpha,
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

    Each figure's suptitle is its source name followed by the observation key
    (``"S1_tdc_off - S1:Image:ArrayData"``), since sources typically observe the
    same screen. Pass ``title_prefix`` to override it, or ``title=False`` to drop
    the suptitles.

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

    Each figure's suptitle is its source name followed by the observation key
    (``"S1_tdc_off - S1:Image:ArrayData"``), since sources typically observe the
    same screen. Pass ``title_prefix`` to override it, or ``title=False`` to drop
    the suptitles.

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
