#!/usr/bin/env python3
"""
Инференс DirectComptonModel для произвольных входных параметров (a0, tau, theta).

Загружает обученную модель и нормализатор, строит график True vs Predicted
в том же стиле, что и визуализация валидации (NN_for_averaging_full_param_angles_pi_2_inference_time.py).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

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
    DATASET_NUM_SAMPLES,
    DATASET_TIMESTEPS,
    PLOT_LABEL_FS,
    PLOT_TEXT_FS,
    PLOT_TICK_FS,
    PLOT_VIS_FONT_SCALE,
    model_file,
    normalizer_file,
)
from compton_averaging.model import DirectComptonModel  # noqa: E402
from compton_averaging.normalizer import DataNormalizer  # noqa: E402
from compton_averaging.physics import init_spectrum_sqinrp  # noqa: E402
from compton_averaging.plotting import (  # noqa: E402
    spectrum_peak_zoom_region,
    spectrum_xlim_skip_leading_zeros,
)
from src.calc_spectrum import (  # noqa: E402
    D2ECalculator_angles,
    calculate_N_photons_with_freqs,
    calculate_trajectory,
)

plt.style.use("physrev.mplstyle")
plt.rcParams["figure.dpi"] = "300"

GAMMA = 1.0

MODEL_FILE = model_file(BASE_DIR)
NORMALIZER_FILE = normalizer_file(BASE_DIR)

TARGET_FREQS = np.linspace(0.01, 2.0, DATASET_TIMESTEPS, dtype=np.float32)


def load_model_and_normalizer(device: torch.device) -> tuple[DirectComptonModel, DataNormalizer]:
    if not os.path.exists(NORMALIZER_FILE):
        raise FileNotFoundError(
            f"Файл нормализатора не найден: {NORMALIZER_FILE}\n"
            "Сначала обучите модель или укажите путь через --normalizer."
        )
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_FILE}\n"
            "Сначала обучите модель или укажите путь через --model."
        )

    print(f"Загрузка нормализатора из {NORMALIZER_FILE}...")
    with open(NORMALIZER_FILE, "rb") as f:
        normalizer = pickle.load(f)
    normalizer.to(device)

    print(f"Загрузка модели из {MODEL_FILE}...")
    model = DirectComptonModel()
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model, normalizer


def compute_ground_truth_spectrum(
    a0: float,
    dur: float,
    theta: float,
    timesteps: int = DATASET_TIMESTEPS,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Эталонный спектр на сетке target_freqs (как в ComptonDataset)."""
    target_freqs = np.linspace(0.01, 2.0, timesteps, dtype=np.float32)

    if a0**2 * dur >= 5:
        arr_chi = target_freqs.copy()
        pm = gamma + np.sqrt(gamma**2 - 1)
        arr_theta = np.array([theta], dtype=np.float64)
        grid_chi, grid_theta = np.meshgrid(arr_chi, arr_theta, indexing="ij")

        calc = D2ECalculator_angles(a0=a0 / np.sqrt(2), T=dur, gamma=gamma)
        k = calc.k(grid_theta)
        b = calc.b(grid_theta)
        chi_bar = 1.0 / k
        xi = 1.0 / (k + a0**2 * b)
        y = 0.5
        chi0 = chi_bar - (chi_bar - xi) * y

        d2E = calc.d2E(grid_chi, grid_theta, chi0)
        y_target = (d2E / grid_chi / pm**2).T[-1]
        freqs = arr_chi
    elif a0 >= 0.1:
        t = np.linspace(-100 * 2 * np.pi, 100 * 2 * np.pi, timesteps)
        state = calculate_trajectory([0.0] * 6, t, a0=a0, duration=dur)
        y_target, freqs_raw = calculate_N_photons_with_freqs(state, t, theta=theta)
        y_target = np.interp(target_freqs, freqs_raw, y_target, left=0.0, right=0.0)
        freqs = target_freqs.copy()
    else:
        t = np.linspace(-50 * 2 * np.pi, 50 * 2 * np.pi, timesteps)
        state = calculate_trajectory([0.0] * 6, t, a0=a0, duration=dur)
        y_target, freqs_raw = calculate_N_photons_with_freqs(state, t, theta=theta)
        y_target = np.interp(target_freqs, freqs_raw, y_target, left=0.0, right=0.0)
        freqs = target_freqs.copy()

    y_target = np.asarray(y_target, dtype=np.float32)
    y_target = np.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
    y_target = np.clip(y_target, a_min=0.0, a_max=None)
    return np.asarray(freqs, dtype=np.float32), y_target


def predict_spectrum(
    model: DirectComptonModel,
    normalizer: DataNormalizer,
    a0: float,
    dur: float,
    theta: float,
    device: torch.device,
) -> np.ndarray:
    p_test = torch.tensor([a0, dur, theta], dtype=torch.float32)
    p_norm = normalizer.norm_params(p_test.unsqueeze(0).to(device))
    with torch.no_grad():
        y_pred_norm = model(p_norm)
        y_pred_log = normalizer.unnorm_spec(y_pred_norm).cpu().numpy().squeeze()
    y_pred_lin = np.clip(10**y_pred_log - 1e-9, a_min=0.0, a_max=None)
    return y_pred_lin.astype(np.float32)


def compute_reference_on_freq_grid(
    freq_axis: np.ndarray,
    a0: float,
    dur: float,
    theta: float,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Численный спектр snlcs, интерполированный на freq_axis (для xlim, как в основном скрипте)."""
    pm = gamma + np.sqrt(gamma**2 - 1)
    spectrum_sqinrp = init_spectrum_sqinrp(BASE_DIR, a0, dur, gamma)

    arr_chi = np.linspace(0.01, 2.0, 10000)
    arr_phi = np.array([0.0], dtype=np.float64)
    arr_theta = np.array([theta], dtype=np.float64)

    spec_sqintrp, _ = spectrum_sqinrp.get_spectrum(arr_phi, arr_theta, arr_chi)
    y_real = np.asarray(spec_sqintrp[0][0] / arr_chi / pm**2, dtype=np.float32).squeeze()
    return np.interp(freq_axis, arr_chi, y_real, left=0.0, right=0.0)


def resolve_zoom_region(
    freq_axis: np.ndarray,
    y_true_lin: np.ndarray,
    zoom_xmin: float | None = None,
    zoom_xmax: float | None = None,
    zoom_ymin: float | None = None,
    zoom_ymax: float | None = None,
) -> tuple[float, float, float, float]:
    """Границы zoom box: авто через spectrum_peak_zoom_region или ручные переопределения."""
    auto_xmin, auto_xmax, auto_ymin, auto_ymax = spectrum_peak_zoom_region(freq_axis, y_true_lin)

    xmin = auto_xmin if zoom_xmin is None else float(zoom_xmin)
    xmax = auto_xmax if zoom_xmax is None else float(zoom_xmax)
    ymin = auto_ymin if zoom_ymin is None else float(zoom_ymin)
    ymax = auto_ymax if zoom_ymax is None else float(zoom_ymax)

    if xmax <= xmin:
        raise ValueError(f"zoom-xmax ({xmax}) должно быть больше zoom-xmin ({xmin})")
    if ymax <= ymin:
        raise ValueError(f"zoom-ymax ({ymax}) должно быть больше zoom-ymin ({ymin})")

    return xmin, xmax, ymin, ymax


def save_spectrum_plot(
    freq_axis: np.ndarray,
    y_true_lin: np.ndarray,
    y_pred_lin: np.ndarray,
    a0: float,
    dur: float,
    theta: float,
    output_path: str,
    y_real_on_freq: np.ndarray | None = None,
    zoom_xmin: float | None = None,
    zoom_xmax: float | None = None,
    zoom_ymin: float | None = None,
    zoom_ymax: float | None = None,
) -> float:
    """Сохраняет PNG в формате валидационной визуализации. Возвращает ошибку по площади (%)."""
    vis_tick_fs = PLOT_TICK_FS * PLOT_VIS_FONT_SCALE
    vis_label_fs = PLOT_LABEL_FS * PLOT_VIS_FONT_SCALE
    vis_text_fs = PLOT_TEXT_FS * PLOT_VIS_FONT_SCALE
    vis_footer_fs = 16.5 * PLOT_VIS_FONT_SCALE

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_true = float(_trapz(y_true_lin, x=freq_axis))
    area_pred = float(_trapz(y_pred_lin, x=freq_axis))
    area_err_pct = 100.0 * abs(area_pred - area_true) / (area_true + 1e-8)

    spectra_for_xlim = (y_true_lin, y_pred_lin)
    if y_real_on_freq is not None:
        spectra_for_xlim = (*spectra_for_xlim, y_real_on_freq)

    x_min, x_max = spectrum_xlim_skip_leading_zeros(freq_axis, *spectra_for_xlim)
    zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = resolve_zoom_region(
        freq_axis,
        y_true_lin,
        zoom_xmin=zoom_xmin,
        zoom_xmax=zoom_xmax,
        zoom_ymin=zoom_ymin,
        zoom_ymax=zoom_ymax,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        freq_axis,
        y_true_lin,
        label=r"True $\frac{d^2N}{d\omega d\Omega}$ Envelope",
        color="red",
        linestyle="-",
        linewidth=1.5,
    )
    ax.plot(
        freq_axis,
        y_pred_lin,
        label=r"Predicted $\frac{d^2N}{d\omega d\Omega}$ Envelope",
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    ax.set_xlim(x_min, x_max)

    axins = inset_axes(ax, width="38%", height="32%", loc="upper right", borderpad=1.2)
    axins.plot(freq_axis, y_true_lin, color="red", linestyle="-", linewidth=1.5)
    axins.plot(freq_axis, y_pred_lin, color="blue", linestyle="--", linewidth=2)
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

    ax.set_xlabel(r"$y$", fontsize=vis_label_fs, labelpad=0)
    ax.set_ylabel(r"$\frac{d^2N}{d\omega d\Omega}$", fontsize=vis_label_fs)
    ax.legend(fontsize=vis_text_fs, loc="center right")
    ax.grid(True, alpha=0.3)

    info_text = (
        r"$\mathrm{Input\ params:}$" + "\n"
        f"$a_0$ = {a0:.3f}\n"
        f"$\\tau$ = {(dur / np.pi):.1f}$\\pi$\n"
        f"$\\theta$ = $\\pi - {(np.pi - theta):.3f}$"
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
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "gray",
            "linewidth": 1.2,
            "boxstyle": "square,pad=0.5",
        },
        ha="left",
        va="bottom",
        clip_on=True,
        zorder=10,
    )

    fig.subplots_adjust(right=0.95, bottom=0.22)
    ax.text(
        0.5,
        -0.19,
        f"Total photon count error: {area_err_pct:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=vis_footer_fs,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray", "pad": 6},
        clip_on=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return area_err_pct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Предсказание спектра Compton по входным параметрам a0, tau, theta.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a0", type=float, default=None, help="Нормированная амплитуда поля a0")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Длительность импульса tau (в единицах pi): dur = tau * pi",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Угол theta в радианах (диапазон обучения: pi/2 .. pi)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE,
        help="Путь к файлу весов модели (.pth)",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default=NORMALIZER_FILE,
        help="Путь к файлу нормализатора (.pkl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь для сохранения PNG (по умолчанию: prediction_a0=..._tau=..._theta=....png)",
    )
    parser.add_argument(
        "--skip-snlcs",
        action="store_true",
        help="Не вызывать snlcs для уточнения xlim (быстрее, если пакет недоступен)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Запросить параметры интерактивно, если не заданы через CLI",
    )
    zoom = parser.add_argument_group(
        "zoom box",
        "Ручные границы увеличенной области (inset). "
        "Не указанные границы определяются автоматически.",
    )
    zoom.add_argument(
        "--zoom-xmin",
        type=float,
        default=None,
        metavar="FREQ",
        help="Левая граница zoom box по freq_axis",
    )
    zoom.add_argument(
        "--zoom-xmax",
        type=float,
        default=None,
        metavar="FREQ",
        help="Правая граница zoom box по freq_axis",
    )
    zoom.add_argument(
        "--zoom-ymin",
        type=float,
        default=None,
        metavar="Y",
        help="Нижняя граница zoom box по значению y",
    )
    zoom.add_argument(
        "--zoom-ymax",
        type=float,
        default=None,
        metavar="Y",
        help="Верхняя граница zoom box по значению y",
    )
    return parser.parse_args()


def prompt_float(name: str, default: float) -> float:
    raw = input(f"{name} [{default}]: ").strip()
    return float(raw) if raw else default


def main() -> None:
    global MODEL_FILE, NORMALIZER_FILE

    args = parse_args()
    MODEL_FILE = args.model
    NORMALIZER_FILE = args.normalizer

    a0 = args.a0
    tau = args.tau
    theta = args.theta

    need_interactive = args.interactive or a0 is None or tau is None or theta is None
    if need_interactive:
        print("Введите входные параметры (Enter — значение по умолчанию):")
        if a0 is None:
            a0 = prompt_float("a0", 1.0)
        if tau is None:
            tau = prompt_float("tau (в единицах pi)", 40.0)
        if theta is None:
            theta = prompt_float("theta (рад)", 2.5)

    dur = float(tau) * np.pi
    a0 = float(a0)
    theta = float(theta)

    if not (np.pi / 2 <= theta <= np.pi):
        print(f"Предупреждение: theta={theta:.4f} вне типичного диапазона обучения [pi/2, pi].")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            BASE_DIR,
            f"prediction_a0={a0:.3g}_tau={tau:.3g}pi_theta={theta:.3g}.png",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model, normalizer = load_model_and_normalizer(device)

    print(f"Расчёт эталонного спектра: a0={a0}, tau={tau}*pi, theta={theta:.4f} rad...")
    freq_axis, y_true_lin = compute_ground_truth_spectrum(a0, dur, theta)

    print("Инференс нейросети...")
    y_pred_lin = predict_spectrum(model, normalizer, a0, dur, theta, device)

    y_real_on_freq = None
    if not args.skip_snlcs:
        try:
            print("Расчёт snlcs-спектра для xlim...")
            y_real_on_freq = compute_reference_on_freq_grid(freq_axis, a0, dur, theta)
        except Exception as exc:
            print(f"snlcs недоступен, xlim без численного эталона: {exc}")

    zoom_manual = any(
        v is not None
        for v in (args.zoom_xmin, args.zoom_xmax, args.zoom_ymin, args.zoom_ymax)
    )
    if zoom_manual:
        print(
            "Zoom box: "
            f"x=[{args.zoom_xmin if args.zoom_xmin is not None else 'auto'}, "
            f"{args.zoom_xmax if args.zoom_xmax is not None else 'auto'}], "
            f"y=[{args.zoom_ymin if args.zoom_ymin is not None else 'auto'}, "
            f"{args.zoom_ymax if args.zoom_ymax is not None else 'auto'}]"
        )

    area_err_pct = save_spectrum_plot(
        freq_axis,
        y_true_lin,
        y_pred_lin,
        a0,
        dur,
        theta,
        output_path,
        y_real_on_freq=y_real_on_freq,
        zoom_xmin=args.zoom_xmin,
        zoom_xmax=args.zoom_xmax,
        zoom_ymin=args.zoom_ymin,
        zoom_ymax=args.zoom_ymax,
    )

    print(f"Ошибка по площади: {area_err_pct:.2f}%")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
