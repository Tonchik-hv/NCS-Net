"""Утилиты оформления графиков спектров."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from compton_averaging.config import (
    CBAR_SCALE_EXP,
    PLOT_CBAR_LABEL_FS,
    PLOT_LABEL_FS,
    PLOT_TEXT_FS,
    PLOT_TICK_FS,
)


def spectrum_xlim_skip_leading_zeros(
    freq_axis: np.ndarray,
    *spectra: np.ndarray,
    skip_zero_fraction: float = 0.7,
    zero_rel_tol: float = 1e-8,
) -> tuple[float, float]:
    """Левая граница xlim: отсечь skip_zero_fraction ведущих почти нулевых точек."""
    y_peak = max(float(np.max(s)) for s in spectra if s.size)
    y_ref = max(y_peak, zero_rel_tol)
    y_combined = np.max(np.stack(spectra, axis=0), axis=0)
    is_zero = y_combined <= zero_rel_tol * y_ref

    n_leading = 0
    for z in is_zero:
        if z:
            n_leading += 1
        else:
            break

    skip = int(np.floor(skip_zero_fraction * n_leading))
    i_start = min(skip, len(freq_axis) - 1)
    return float(freq_axis[i_start]), float(freq_axis[-1])


def spectrum_peak_zoom_region(
    freq_axis: np.ndarray,
    y_true_lin: np.ndarray,
    zero_rel_tol: float = 1e-8,
) -> tuple[float, float, float, float]:
    """Zoom-область по y_true_lin: x от x_left до середины ненулевого участка, y от y[i_mid] до y_max."""
    x = np.asarray(freq_axis, dtype=np.float64).squeeze()
    y = np.asarray(y_true_lin, dtype=np.float64).squeeze()
    if y.size == 0 or x.size != y.size:
        return float(x[0]) if x.size else 0.0, float(x[-1]) if x.size else 1.0, 0.0, 1.0

    y_max = float(np.max(y))
    if y_max <= 0.0:
        return float(x[0]), float(x[-1]), 0.0, 1.0

    level = zero_rel_tol * max(y_max, zero_rel_tol)
    nonzero = y > level
    if not np.any(nonzero):
        return float(x[0]), float(x[-1]), 0.0, y_max

    i_left = int(np.argmax(nonzero))
    i_right = int(len(nonzero) - 1 - np.argmax(nonzero[::-1]))
    i_mid = i_left + (i_right - i_left) // 2

    zoom_xmin = float(x[i_left])
    zoom_xmax = float(x[i_mid])
    zoom_ymin = float(y[i_mid])
    zoom_ymax = y_max

    if zoom_xmax <= zoom_xmin:
        zoom_xmax = float(x[min(i_left + 1, len(x) - 1)])
    if zoom_ymax <= zoom_ymin:
        zoom_ymin = 0.5 * zoom_ymax

    return zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax


def style_spectrum_2d_map_axes(
    ax,
    gamma: float,
    xlim=(0.01, 1.5),
    ytick_decimals: int = 1,
    tick_fs: float = PLOT_TICK_FS,
    label_fs: float = PLOT_LABEL_FS,
) -> None:
    """Оформление осей 2D-карт I(ω, θ), как в averaging_origin.ipynb."""
    ax.set_xlabel(r"$y$", fontsize=label_fs)
    ax.set_ylabel(r"$\pi - \theta$", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ticks = ax.get_yticks()
    labels = [f"{t / gamma:.{ytick_decimals}f}" for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.set_ylim(0, None)
    ax.set_xlim(*xlim)


def style_colorbar(
    cbar,
    label: str,
    zmax: float,
    tick_decimals: int | None = 1,
    scale_exp: int | None = None,
    tick_fs: float = PLOT_TICK_FS,
    label_fs: float = PLOT_CBAR_LABEL_FS,
) -> None:
    zmax = float(zmax)
    cbar.set_label(label, fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    cbar.mappable.set_clim(0.0, zmax)
    locator = MaxNLocator(nbins=6, min_n_ticks=3)
    ticks = locator.tick_values(0.0, zmax)
    ticks = ticks[(ticks >= 0.0) & (ticks <= zmax + 1e-12 * max(zmax, 1.0))]
    if ticks.size == 0:
        ticks = np.array([0.0, zmax])

    exp = CBAR_SCALE_EXP if scale_exp is None else scale_exp
    scale = 10.0 ** exp

    tick_labels = []
    for t in ticks:
        scaled = t / scale
        if tick_decimals is None:
            if abs(scaled - round(scaled)) < 0.05:
                tick_labels.append(f"{int(round(scaled))}")
            else:
                tick_labels.append(f"{scaled:.1g}")
        else:
            tick_labels.append(f"{scaled:.{tick_decimals}f}")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels, fontsize=tick_fs)

    if exp != 0:
        cbar.ax.text(
            0.5,
            1.03,
            f"1e{exp}",
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=tick_fs,
        )


def save_val_2d_spectrum_map_png(
    path: str,
    freq_axis: np.ndarray,
    y_pos_1d: np.ndarray,
    z: np.ndarray,
    gamma: float,
    cbar_label: str,
    a0: float,
    dur: float,
    tick_fs: float = PLOT_TICK_FS,
    label_fs: float = PLOT_LABEL_FS,
    cbar_label_fs: float = PLOT_CBAR_LABEL_FS,
    text_fs: float = PLOT_TEXT_FS,
) -> None:
    """Сохранение одной 2D-карты I(ω, θ) в том же стиле, что и основные карты по сетке θ."""
    fig = plt.figure(figsize=(12, 7))
    zmax = float(np.nanmax(z))
    mesh = plt.pcolormesh(
        freq_axis,
        y_pos_1d,
        z,
        shading="gouraud",
        cmap="viridis",
        vmin=0.0,
        vmax=zmax,
    )
    cbar = plt.colorbar(mesh, extend="neither")
    style_colorbar(cbar, cbar_label, zmax, tick_fs=tick_fs, label_fs=cbar_label_fs)
    ax = plt.gca()
    style_spectrum_2d_map_axes(ax, float(gamma), xlim=(0.01, 1.5), tick_fs=tick_fs, label_fs=label_fs)
    plt.grid(False)
    ax.text(
        0.02,
        0.98,
        f"$a_0$ = {a0:.4g}\n$\\tau$ = {(dur / np.pi):.4g}$\\pi$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=text_fs,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
        clip_on=False,
    )
    plt.subplots_adjust(right=0.82, bottom=0.12)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
