import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from compton_averaging.config import model_file, normalizer_file
from compton_averaging.model import DirectComptonModel
from compton_averaging.normalizer import DataNormalizer
from snlcs.external_fields import ExternalField
from snlcs.ebeam import eGaussBeam
from src.calc_spectrum import D2ECalculator_angles
from snlcs.solvers import Spectrum
from snlcs.methods.motion import MotionIntegrationMethod
from snlcs.methods.spectrum import SpectrumMethod
import time

plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "300"

PLOT_TICK_FS = 16
PLOT_LABEL_FS = 20
PLOT_CBAR_LABEL_FS = 18
PLOT_TEXT_FS = 14
CBAR_SCALE_EXP = 7
PLOT_VIS_FONT_SCALE = 1.2 * 1.15


def style_map_axes(ax, gamma: float, xlim, y_max=None, ytick_decimals: int = 3, tick_fs: float = PLOT_TICK_FS) -> None:
    ax.tick_params(axis="both", labelsize=tick_fs)
    ticks = ax.get_yticks()
    labels = [f"{t / gamma:.{ytick_decimals}f}" for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, None)
    ax.set_xlim(*xlim)


def style_colorbar(
    cbar,
    label: str,
    zmax: float,
    tick_decimals: int = 1,
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

    exp = CBAR_SCALE_EXP
    scale = 10.0 ** exp

    tick_labels = [f"{t / scale:.{tick_decimals}f}" for t in ticks]
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


def add_beam_params_text(ax, a0: float, dur: float, text_fs: float = PLOT_TEXT_FS) -> None:
    ax.text(
        0.98,
        0.98,
        f"$a_0$ = {a0:.1f}\n$\\tau$ = {(dur / np.pi):.1f}$\\pi$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=text_fs,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
        clip_on=False,
    )


def integrate_d2n_omega_solid_angle(I, theta_axis, freq_axis):
    """∫∫ (d²N/dωdΩ)·2π sin θ dω dθ; θ сортируется по возрастанию для корректного знака."""
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    theta_1d = np.asarray(theta_axis, dtype=np.float64)
    freq_1d = np.asarray(freq_axis, dtype=np.float64)
    sort_idx = np.argsort(theta_1d)
    theta_sorted = theta_1d[sort_idx]
    I_sorted = np.asarray(I, dtype=np.float64)[sort_idx, :]
    jac = 2.0 * np.pi * np.sin(theta_sorted)[:, np.newaxis]
    return float(_trapz(_trapz(I_sorted * jac, freq_1d, axis=1), theta_sorted))




def main():
    # 1. Load parameters and sample beam
    path_extf_params = os.path.join(BASE_DIR, "config/ExtFieldsParams.yaml")
    path_ebeam_params = os.path.join(BASE_DIR, "config/eBeamParams.yaml")

    extfields_params = ExternalField.from_yaml(path_extf_params)
    ebeam_params = eGaussBeam.from_yaml(path_ebeam_params)
    in_states = ebeam_params.initial_states
    N_particles = in_states.shape[0]

    a0_lab = extfields_params.a0
    dur_lab = extfields_params.T

    # 2. Load NN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(normalizer_file(BASE_DIR), "rb") as f:
        normalizer = pickle.load(f)
    normalizer.to(device)

    model = DirectComptonModel(params_dim=3).to(device)
    checkpoint = torch.load(model_file(BASE_DIR), map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 3. Define common LAB grid (same bounds as snlcs/src/test.ipynb)
    gamma0_mean = ebeam_params.gamma
    freq_i = 0.01
    freq_f = 1.5
    freq_N = 1000
    theta_i = 0.0 / gamma0_mean
    theta_f = 2.0 / gamma0_mean
    theta_N = 100

    chi_lab_grid = np.linspace(freq_i, freq_f, freq_N)
    theta_lab_grid = np.pi - np.linspace(theta_i, theta_f, theta_N)

    CHI_LAB, THETA_LAB = np.meshgrid(chi_lab_grid, theta_lab_grid, indexing="xy")
    D2N_LAB_TOTAL = np.zeros_like(CHI_LAB)

    freq_axis_2d = np.linspace(0.01, 2.0, 2500).astype(np.float32)

    print("Computing first picture (NN) on CPU...")
    t0_nn = time.perf_counter()
    for i in range(N_particles):
        px, py, pz = in_states[i, 3], in_states[i, 4], in_states[i, 5]
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        gamma_i = np.sqrt(1 + p_mag**2)
    
        beta_vec = np.array([px, py, pz]) / gamma_i
    
        nx_obs_1d = np.sin(theta_lab_grid)
        ny_obs_1d = np.zeros_like(theta_lab_grid)
        nz_obs_1d = np.cos(theta_lab_grid)
        n_lab = np.stack([nx_obs_1d, ny_obs_1d, nz_obs_1d], axis=-1)
    
        beta_dot_n = np.dot(n_lab, beta_vec)
        D_obs = gamma_i * (1.0 - beta_dot_n)
    
        n_prime = (n_lab - gamma_i * beta_vec + (gamma_i**2 / (gamma_i + 1.0)) * np.outer(beta_dot_n, beta_vec)) / D_obs[:, None]
    
        k_las = np.array([0.0, 0.0, 1.0])
        beta_dot_k = np.dot(beta_vec, k_las)
        D_las = gamma_i * (1.0 - beta_dot_k)
        k_prime = (k_las - gamma_i * beta_vec + (gamma_i**2 / (gamma_i + 1.0)) * beta_dot_k * beta_vec) / D_las
    
        cos_theta_prime_1d = np.sum(n_prime * k_prime, axis=-1)
        cos_theta_prime_1d = np.clip(cos_theta_prime_1d, -1.0, 1.0)
        theta_prime_1d = np.arccos(cos_theta_prime_1d).astype(np.float32)
    
        p_grid = np.stack([
            np.full_like(theta_prime_1d, a0_lab),
            np.full_like(theta_prime_1d, dur_lab),
            theta_prime_1d
        ], axis=1)
    
        with torch.no_grad():
            p_grid_t = torch.from_numpy(p_grid).to(device)
            p_grid_norm = normalizer.norm_params(p_grid_t)
            y_grid_norm = model(p_grid_norm)
            y_grid_log = normalizer.unnorm_spec(y_grid_norm).squeeze(1)
            # y_grid_lin has shape (theta_N, 2500)
            y_grid_lin = torch.clamp(10 ** y_grid_log - 1e-9, min=0.0).cpu().numpy()
        
        D_obs_2d = D_obs[:, None]
        chi_prime = (D_obs_2d / D_las) * 4.0 * gamma0_mean**2 * CHI_LAB
    
        # y_grid_lin: (theta_N, 2500); chi_prime: (theta_N, freq_N)
        dx = freq_axis_2d[1] - freq_axis_2d[0]
        idx = (chi_prime - freq_axis_2d[0]) / dx
    
        idx_floor = np.floor(idx).astype(int)
        idx_ceil = idx_floor + 1
        weight_ceil = idx - idx_floor
        weight_floor = 1.0 - weight_ceil
    
        valid = (idx_floor >= 0) & (idx_ceil < len(freq_axis_2d))
    
        intensity_prime = np.zeros_like(chi_prime)
        row_idx = np.broadcast_to(np.arange(len(theta_lab_grid))[:, None], CHI_LAB.shape)
    
        intensity_prime[valid] = (
            y_grid_lin[row_idx[valid], idx_floor[valid]] * weight_floor[valid] +
            y_grid_lin[row_idx[valid], idx_ceil[valid]] * weight_ceil[valid]
        )
    
        jacobian_1d = 4.0 * gamma0_mean**2 / (D_obs * D_las)
        jacobian_full = np.broadcast_to(jacobian_1d[:, None], CHI_LAB.shape)
    
        intensity_lab = intensity_prime * jacobian_full
    
        D2N_LAB_TOTAL += intensity_lab

    # D2N_LAB_TOTAL /= N_particles
    time_nn = time.perf_counter() - t0_nn
    print(f"Time for first picture (NN) on CPU: {time_nn:.3f} s")

    vis_tick_fs = PLOT_TICK_FS * PLOT_VIS_FONT_SCALE
    vis_label_fs = PLOT_LABEL_FS * PLOT_VIS_FONT_SCALE
    vis_text_fs = PLOT_TEXT_FS * PLOT_VIS_FONT_SCALE
    vis_cbar_label_fs = PLOT_CBAR_LABEL_FS * PLOT_VIS_FONT_SCALE

    plt.figure(figsize=(12, 7))
    y_pos_lab = gamma0_mean * (np.pi - THETA_LAB)

    zmax_nn = float(np.nanmax(D2N_LAB_TOTAL*0.599))
    mesh = plt.pcolormesh(
        CHI_LAB, y_pos_lab, D2N_LAB_TOTAL*0.599, shading="gouraud", cmap="viridis", vmin=0.0, vmax=zmax_nn
    )

    plt.xlabel(r"$y_{\text{lab}}$", fontsize=vis_label_fs)
    plt.ylabel(r"$\gamma_{\text{mean}}(\pi - \theta_{\text{lab}})$", fontsize=vis_label_fs)

    cbar = plt.colorbar(mesh, extend="neither")
    style_colorbar(
        cbar,
        r"Predicted $\frac{d^2N}{d\omega_{\text{lab}} d\Omega_{\text{lab}}}$ Envelope",
        zmax_nn,
        tick_fs=vis_tick_fs,
        label_fs=vis_cbar_label_fs,
    )

    ax2d = plt.gca()
    ax2d.set_facecolor(plt.cm.viridis(0))
    style_map_axes(
        ax2d,
        gamma0_mean,
        (freq_i, freq_f),
        y_max=np.nanmax(y_pos_lab),
        ytick_decimals=3,
        tick_fs=vis_tick_fs,
    )

    add_beam_params_text(ax2d, a0_lab, dur_lab, text_fs=vis_text_fs)

    plt.subplots_adjust(right=0.82, bottom=0.12)
    out_file = "predicted_beam_spectrum_map.png"
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved to {out_file}")

    # 4. Analytical beam-averaged spectrum (D2ECalculator_angles, as in test.ipynb)
    nx = np.sin(THETA_LAB)
    ny = 0.0
    nz = np.cos(THETA_LAB)

    d2N_analytic_avg = np.zeros_like(CHI_LAB)

    print("Computing second picture (Analytical) on CPU...")
    t0_analytic = time.perf_counter()
    nx_obs_1d = np.sin(theta_lab_grid)
    ny_obs_1d = np.zeros_like(theta_lab_grid)
    nz_obs_1d = np.cos(theta_lab_grid)
    n_lab = np.stack([nx_obs_1d, ny_obs_1d, nz_obs_1d], axis=-1)
    k_las = np.array([0.0, 0.0, 1.0])

    for i in range(N_particles):
        px, py, pz = in_states[i, 3], in_states[i, 4], in_states[i, 5]
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        gamma_i = np.sqrt(1 + p_mag**2)
        beta_vec = np.array([px, py, pz]) / gamma_i
    
        beta_dot_n = np.dot(n_lab, beta_vec)
        D_obs = gamma_i * (1.0 - beta_dot_n)
        n_prime = (n_lab - gamma_i * beta_vec + (gamma_i**2 / (gamma_i + 1.0)) * np.outer(beta_dot_n, beta_vec)) / D_obs[:, None]
    
        beta_dot_k = np.dot(beta_vec, k_las)
        D_las = gamma_i * (1.0 - beta_dot_k)
        k_prime = (k_las - gamma_i * beta_vec + (gamma_i**2 / (gamma_i + 1.0)) * beta_dot_k * beta_vec) / D_las
    
        cos_theta_prime = np.sum(n_prime * k_prime, axis=-1)
        cos_theta_prime = np.clip(cos_theta_prime, -1.0, 1.0)
        theta_prime = np.arccos(cos_theta_prime)
    
        D_obs_2d = D_obs[:, None]
        chi_prime_grid = (D_obs_2d / D_las) * 4.0 * gamma0_mean**2 * CHI_LAB
        theta_prime_2d = np.broadcast_to(theta_prime[:, None], CHI_LAB.shape)
    
        # Инициализация аналитического калькулятора в системе отсчета покоя электрона
        calc_analytic_rf = D2ECalculator_angles(a0_lab / np.sqrt(2), dur_lab, 1.0)

        k = calc_analytic_rf.k(theta_prime_2d)
        b = calc_analytic_rf.b(theta_prime_2d)
        chi_bar = 1.0 / k
        xi = 1.0 / (k + (a0_lab / np.sqrt(2)) ** 2 * b)
        chi0 = chi_bar - (chi_bar - xi) * 0.5

        d2E_prime = calc_analytic_rf.d2E(chi_prime_grid, theta_prime_2d, chi0)
    
        with np.errstate(divide='ignore', invalid='ignore'):
            d2N_prime = d2E_prime / chi_prime_grid
            d2N_prime = np.nan_to_num(d2N_prime)

        jacobian_1d = 4.0 * gamma0_mean**2 / (D_obs * D_las)
        jacobian_full = np.broadcast_to(jacobian_1d[:, None], CHI_LAB.shape)
    
        d2N_p = d2N_prime * jacobian_full
        d2N_analytic_avg += d2N_p

    # d2N_analytic_avg /= N_particles
    time_analytic = time.perf_counter() - t0_analytic
    print(f"Time for second picture (Analytical) on CPU: {time_analytic:.3f} s")

    # Относительная ошибка интеграла predicted vs analytical (как rel_int_samples в валидации)
    def integrate_d2n_omega_solid_angle(I, theta_axis, freq_axis):
        """∫∫ (d²N/dωdΩ)·2π sin θ dω dθ; θ сортируется по возрастанию для корректного знака."""
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        theta_1d = np.asarray(theta_axis, dtype=np.float64)
        freq_1d = np.asarray(freq_axis, dtype=np.float64)
        sort_idx = np.argsort(theta_1d)
        theta_sorted = theta_1d[sort_idx]
        I_sorted = np.asarray(I, dtype=np.float64)[sort_idx, :]
        jac = 2.0 * np.pi * np.sin(theta_sorted)[:, np.newaxis]
        return float(_trapz(_trapz(I_sorted * jac, freq_1d, axis=1), theta_sorted))

    eps_i = 1e-30
    ip = integrate_d2n_omega_solid_angle(D2N_LAB_TOTAL, theta_lab_grid, chi_lab_grid)
    ia = integrate_d2n_omega_solid_angle(d2N_analytic_avg, theta_lab_grid, chi_lab_grid)
    rel_int_pct = (
        100.0 * abs(ip - ia) / (abs(ia) + eps_i)
        if np.isfinite(ip) and np.isfinite(ia)
        else float("nan")
    )
    print(
        f"\n--- 2D beam-averaged: ∫∫ (d²N/dωdΩ)·2π sin θ dω dθ (predicted vs analytical) ---\n"
        f"  Предсказание (NN):     {ip:.6e}\n"
        f"  Аналитика:             {ia:.6e}\n"
        f"  Относительная ошибка:  {rel_int_pct:.4f} %"
    )

    plt.figure(figsize=(12, 7))
    y_pos = gamma0_mean * (np.pi - THETA_LAB)
    zmax_an = float(np.nanmax(d2N_analytic_avg))
    mesh = plt.pcolormesh(
        CHI_LAB, y_pos, d2N_analytic_avg/2, shading="gouraud", cmap="viridis", vmin=0.0, vmax=zmax_an
    )
    plt.xlabel(r"$y_{\text{lab}}$", fontsize=vis_label_fs)
    plt.ylabel(r"$\gamma_{\text{mean}}(\pi - \theta_{\text{lab}})$", fontsize=vis_label_fs)

    cbar = plt.colorbar(mesh, extend="neither")
    style_colorbar(
        cbar,
        r"Analytical $\frac{d^2N}{d\omega_{\text{lab}} d\Omega_{\text{lab}}}$ Envelope",
        zmax_an,
        tick_fs=vis_tick_fs,
        label_fs=vis_cbar_label_fs,
    )

    ax = plt.gca()
    style_map_axes(
        ax, gamma0_mean, (freq_i, freq_f), y_max=np.nanmax(y_pos), ytick_decimals=3, tick_fs=vis_tick_fs
    )

    add_beam_params_text(ax, a0_lab, dur_lab, text_fs=vis_text_fs)

    plt.subplots_adjust(right=0.82, bottom=0.12)
    out_file_analytic = "analytical_beam_spectrum_map.png"
    plt.savefig(out_file_analytic, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved to {out_file_analytic}")

    # 5. Third 2D picture (from test.ipynb) without Lorentz transform, coherent=False
    print("Computing third picture on CPU...")
    z0 = ebeam_params.z0
    n_lambda = 2 * abs(z0) / np.pi
    points_per_lambda_sqintrp = 100
    N_sqintrp = int(round(n_lambda * points_per_lambda_sqintrp)) + 1
    arr_t_sqintrp = np.linspace(-2*z0, z0, N_sqintrp)

    phi_i = 0
    phi_f = 1
    phi_N = 1
    arr_phi = np.linspace(phi_i, 2*np.pi*phi_f, phi_N)

    arr_theta_2d = np.pi - np.linspace(theta_i, theta_f, theta_N) 
    arr_freq_2d = np.linspace(freq_i, freq_f, freq_N) * 4 * gamma0_mean ** 2

    spectrum_sqinrp_light = Spectrum(
        ext_field=extfields_params, 
        initial_states=in_states,
        arr_t=arr_t_sqintrp, 
        solver_method=MotionIntegrationMethod.BORIS, 
        spec_calc_method=SpectrumMethod.QUAD_INTERP_LIGHT,
        coherent=False,
    )

    t0_cpu = time.perf_counter()
    spec_sqintrp_2d, _ = spectrum_sqinrp_light.get_spectrum(arr_phi, arr_theta_2d, arr_freq_2d)
    time_cpu = time.perf_counter() - t0_cpu
    print(f"Time for third picture on CPU: {time_cpu:.3f} s")

    gamma = ebeam_params.gamma

    # arr_freq_2d is freq_N, arr_theta_2d is theta_N
    grid_chi, grid_theta = np.meshgrid(arr_freq_2d / (4 * gamma**2), arr_theta_2d, indexing="xy")

    d2N = spec_sqintrp_2d[0] / grid_chi # shape (theta_N, freq_N)

    i_quad = integrate_d2n_omega_solid_angle(d2N, arr_theta_2d, grid_chi[0, :])
    print(
        f"\n--- 2D beam-averaged: ∫∫ (d²N/dωdΩ)·2π sin θ dω dθ (quad-interp, третья картинка) ---\n"
        f"  Quad-interp:           {i_quad:.6e}"
    )

    plt.figure(figsize=(12, 7))
    # положительная ось: 0 при θ=π, растёт к меньшим θ
    y_pos = gamma * (np.pi - grid_theta)
    zmax_quad = float(np.nanmax(d2N))
    mesh = plt.pcolormesh(
        grid_chi, y_pos, d2N, shading="gouraud", cmap="viridis", vmin=0.0, vmax=zmax_quad
    )
    plt.xlabel(r"$y_{\text{lab}}$", fontsize=vis_label_fs)
    plt.ylabel(r"$\gamma_{\text{mean}}(\pi - \theta_{\text{lab}})$", fontsize=vis_label_fs)
    cbar = plt.colorbar(mesh, extend="neither")
    style_colorbar(
        cbar,
        r"Original $\frac{d^2N}{d\omega d\Omega}$",
        zmax_quad,
        tick_fs=vis_tick_fs,
        label_fs=vis_cbar_label_fs,
    )

    ax = plt.gca()
    style_map_axes(
        ax, gamma, (freq_i, freq_f), y_max=np.nanmax(y_pos), ytick_decimals=2, tick_fs=vis_tick_fs
    )

    add_beam_params_text(ax, a0_lab, dur_lab, text_fs=vis_text_fs)

    plt.subplots_adjust(right=0.82, bottom=0.12)
    out_file_third = "analytical_beam_spectrum_theta_frequency_map_lab.png"
    plt.savefig(out_file_third, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved to {out_file_third}")


if __name__ == "__main__":
    main()
