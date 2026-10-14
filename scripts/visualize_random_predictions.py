#!/usr/bin/env python3
"""Визуализация N случайных результатов на валидационном датасете."""
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from compton_averaging.config import (  # noqa: E402
    N_VIS_RESULTS,
    PLOT_LABEL_FS,
    PLOT_TEXT_FS,
    PLOT_TICK_FS,
    PLOT_VIS_FONT_SCALE,
)
from compton_averaging.io import load_for_inference  # noqa: E402
from compton_averaging.physics import init_spectrum_sqinrp  # noqa: E402
from compton_averaging.plotting import (  # noqa: E402
    spectrum_peak_zoom_region,
    spectrum_xlim_skip_leading_zeros,
)

plt.style.use('physrev.mplstyle')
plt.rcParams['figure.dpi'] = "300"


def main():
    device, model, normalizer, val_data = load_for_inference(BASE_DIR)

    # --- ВИЗУАЛИЗАЦИЯ N СЛУЧАЙНЫХ РЕЗУЛЬТАТОВ ---
    n_vis = max(1, int(N_VIS_RESULTS))
    n_vis = min(n_vis, len(val_data))
    if len(val_data) == 0:
        print("Валидационный датасет пуст, визуализация пропущена.")
    else:
        vis_tick_fs = PLOT_TICK_FS * PLOT_VIS_FONT_SCALE
        vis_label_fs = PLOT_LABEL_FS * PLOT_VIS_FONT_SCALE
        vis_text_fs = PLOT_TEXT_FS * PLOT_VIS_FONT_SCALE
        vis_footer_fs = 16.5 * PLOT_VIS_FONT_SCALE
        vis_indices = random.sample(range(len(val_data)), k=n_vis)
        spectrum_times = []
        for k, idx in enumerate(vis_indices, start=1):
            p_test, y_true, freqs = val_data[idx]

            p_norm = normalizer.norm_params(p_test.unsqueeze(0).to(device))
            with torch.no_grad():
                y_pred_norm = model(p_norm)
                y_pred_log = normalizer.unnorm_spec(y_pred_norm).cpu().numpy().squeeze()

            freq_axis = freqs.numpy().squeeze()
            y_true_lin = np.clip(10**y_true.numpy().squeeze() - 1e-9, a_min=0.0, a_max=None)
            y_pred_lin = np.clip(10**y_pred_log - 1e-9, a_min=0.0, a_max=None)

            y_pred_lin = np.clip(y_pred_lin, a_min=0.0, a_max=None)

            # Ошибка по площади под кривой для выбранного сэмпла (в %)
            area_true_sample = np.trapz(y_true_lin, x=freq_axis)
            area_pred_sample = np.trapz(y_pred_lin, x=freq_axis)
            area_err_pct_sample = 100.0 * np.abs(area_pred_sample - area_true_sample) / (area_true_sample + 1e-8)

            a0 = float(p_test[0].item())
            dur = float(p_test[1].item())
            theta = float(p_test[2].item())
            gamma = 1.0
            pm = gamma + np.sqrt(gamma**2-1)

            spectrum_sqinrp = init_spectrum_sqinrp(BASE_DIR, a0, dur, gamma)

            chi_i = 0.01
            chi_f = 2.0
            chi_N = 10000
            arr_chi = np.linspace(chi_i, chi_f, chi_N)
            arr_freq = arr_chi # * 4 * gamma0 ** 2
            arr_phi = np.array([0.0], dtype=np.float64)
            arr_theta = np.array([theta], dtype=np.float64)

            t_spec_start = time.perf_counter()
            spec_sqintrp, _ = spectrum_sqinrp.get_spectrum(arr_phi, arr_theta, arr_freq)
            spectrum_times.append(time.perf_counter() - t_spec_start)
            y_real = np.asarray(spec_sqintrp[0][0]/arr_chi/pm**2, dtype=np.float32).squeeze()

            y_real = np.asarray(y_real, dtype=np.float32)
            y_real = np.nan_to_num(y_real, nan=0.0, posinf=0.0, neginf=0.0)
            y_real = np.clip(y_real, a_min=0.0, a_max=None)

            y_real_on_freq = np.interp(freq_axis, arr_freq, y_real, left=0.0, right=0.0)

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(freq_axis, y_true_lin, label=r'True $\frac{d^2N}{d\omega d\Omega}$ Envelope', color='red', linestyle='-', linewidth=1.5)
            ax.plot(freq_axis, y_pred_lin, label=r'Predicted $\frac{d^2N}{d\omega d\Omega}$ Envelope', color='blue', linestyle='--', linewidth=2)
            ax.plot(arr_freq, y_real, label=r'Original $\frac{d^2N}{d\omega d\Omega}$', alpha=0.4, color='black', linestyle='-', linewidth=1.5)

            x_min, x_max = spectrum_xlim_skip_leading_zeros(
                freq_axis,
                y_true_lin,
                y_pred_lin,
                y_real_on_freq,
            )
            ax.set_xlim(x_min, x_max)

            zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = spectrum_peak_zoom_region(
                freq_axis,
                y_true_lin,
            )

            axins = inset_axes(ax, width="38%", height="32%", loc="upper right", borderpad=1.2)
            axins.plot(freq_axis, y_true_lin, color='red', linestyle='-', linewidth=1.5)
            axins.plot(freq_axis, y_pred_lin, color='blue', linestyle='--', linewidth=2)
            axins.plot(arr_freq, y_real, alpha=0.4, color='black', linestyle='-', linewidth=1.5)
            axins.set_xlim(zoom_xmin, zoom_xmax)
            axins.set_ylim(zoom_ymin, zoom_ymax)
            axins.tick_params(axis="both", labelsize=vis_tick_fs * 0.75)
            axins.grid(True, alpha=0.3)

            ax.add_patch(
                Rectangle(
                    (zoom_xmin, zoom_ymin),
                    zoom_xmax - zoom_xmin,
                    zoom_ymax - zoom_ymin,
                    fill=False,
                    edgecolor="black",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=5,
                )
            )

            # plt.title("Результат работы нейросети (Прямая Генерация)", fontsize=14)
            ax.set_xlabel(r"$y$", fontsize=vis_label_fs, labelpad=0)
            ax.set_ylabel(r"$\frac{d^2N}{d\omega d\Omega}$", fontsize=vis_label_fs)
            ax.legend(fontsize=vis_text_fs, loc="center right")
            ax.grid(True, alpha=0.3)

            p_in = p_test.numpy()
            info_text = (
                r"$\mathrm{Input\ params:}$" + "\n"
                f"$a_0$ = {p_in[0]:.3f}\n"
                f"$\\tau$ = {(p_in[1]/np.pi):.1f}$\\pi$\n"
                f"$\\theta$ = $\\pi - {(np.pi - p_in[2]):.3f}$"
            )
            ax.tick_params(axis="both", labelsize=vis_tick_fs)
            y_peak_plot = max(float(np.max(y_true_lin)), float(np.max(y_pred_lin)))
            info_trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.72,
                0.04 * y_peak_plot,
                info_text,
                transform=info_trans,
                fontsize=vis_text_fs,
                bbox={
                    'facecolor': 'white',
                    'alpha': 0.8,
                    'edgecolor': 'gray',
                    'linewidth': 1.2,
                    'boxstyle': 'square,pad=0.5',
                },
                ha='left',
                va='bottom',
                clip_on=True,
                zorder=10,
            )

            fig.subplots_adjust(right=0.95, bottom=0.22)
            ax.text(
                0.5,
                -0.19,
                f"Total photon count error: {area_err_pct_sample:.2f}%",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=vis_footer_fs,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'gray', 'pad': 6},
                clip_on=False,
            )
            out_file = f"averaged_prediction_result_angles_pi_2_{k}.png"
            fig.savefig(out_file, bbox_inches='tight', pad_inches=0.25)
            plt.close(fig)
            print(f"Результат {k}/{n_vis} сохранен в {out_file}")

        if spectrum_times:
            print(
                f"Среднее время расчёта спектра (spectrum_sqinrp.get_spectrum): "
                f"{np.mean(spectrum_times)*1000:.3f} мс"
            )


if __name__ == "__main__":
    main()
