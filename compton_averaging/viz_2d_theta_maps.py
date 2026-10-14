"""2D визуализация предсказаний I(theta, omega) и метрики."""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from compton_averaging.config import (
    BASE_DIR,
    N_THETA_MAP,
    N_VAL_2D_MAPS_TO_SAVE,
    N_VAL_2D_METRICS_MAX,
    PLOT_CBAR_LABEL_FS,
    PLOT_LABEL_FS,
    PLOT_TEXT_FS,
    PLOT_TICK_FS,
    PLOT_VIS_FONT_SCALE,
    val_2d_metrics_maps_dir,
)
from compton_averaging.model import DirectComptonModel
from compton_averaging.normalizer import DataNormalizer
from compton_averaging.physics import init_spectrum_sqinrp
from compton_averaging.plotting import (
    save_val_2d_spectrum_map_png,
    style_colorbar,
    style_spectrum_2d_map_axes,
)
from src.calc_spectrum import D2ECalculator_angles


def run_2d_theta_maps(device, model, normalizer, val_data, base_dir=None):
    root = base_dir or BASE_DIR
    VAL_2D_METRICS_MAPS_DIR = val_2d_metrics_maps_dir(root)

    # --- 2D ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ I(theta, omega) ---
    if len(val_data) == 0:
        print("Валидационный датасет пуст, 2D визуализация по theta пропущена.")
    else:
        vis_tick_fs = PLOT_TICK_FS * PLOT_VIS_FONT_SCALE
        vis_label_fs = PLOT_LABEL_FS * PLOT_VIS_FONT_SCALE
        vis_text_fs = PLOT_TEXT_FS * PLOT_VIS_FONT_SCALE
        vis_cbar_label_fs = PLOT_CBAR_LABEL_FS * PLOT_VIS_FONT_SCALE

        p_ref, _, freqs_ref = val_data[0]
        # a0_ref = float(p_ref[0].item())
        # dur_ref = float(p_ref[1].item())
        a0_ref = 0.8
        dur_ref = 50.0*np.pi
        gamma_ref = 1.0

        theta_min = np.pi/2
        theta_max = np.pi
        theta_axis = np.linspace(theta_min, theta_max, int(N_THETA_MAP), dtype=np.float32)
        freq_axis_2d = freqs_ref.numpy().squeeze().astype(np.float32)

        p_grid = np.stack(
            [
                np.full_like(theta_axis, a0_ref, dtype=np.float32),
                np.full_like(theta_axis, dur_ref, dtype=np.float32),
                theta_axis,
            ],
            axis=1,
        )

        with torch.no_grad():
            p_grid_t = torch.from_numpy(p_grid).to(device)
            p_grid_norm = normalizer.norm_params(p_grid_t)
            y_grid_norm = model(p_grid_norm)
            y_grid_log = normalizer.unnorm_spec(y_grid_norm).squeeze(1)
            y_grid_lin = torch.clamp(10 ** y_grid_log - 1e-9, min=0.0).cpu().numpy()

        intensity_map = np.nan_to_num(y_grid_lin, nan=0.0, posinf=0.0, neginf=0.0)

        y_pos = gamma_ref * (np.pi - theta_axis)
        fig2d = plt.figure(figsize=(12, 7))
        zmax_pred = float(np.nanmax(intensity_map))
        mesh = plt.pcolormesh(
            freq_axis_2d,
            y_pos,
            intensity_map,
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=zmax_pred,
        )
        cbar = plt.colorbar(mesh, extend="neither")
        style_colorbar(
            cbar,
            r"Predicted $\frac{d^2N}{d\omega d\Omega}$ Envelope",
            zmax_pred,
            tick_decimals=None,
            scale_exp=0,
            tick_fs=vis_tick_fs,
            label_fs=vis_cbar_label_fs,
        )
        ax2d = plt.gca()
        style_spectrum_2d_map_axes(
            ax2d, float(gamma_ref), xlim=(0.01, 1.5), tick_fs=vis_tick_fs, label_fs=vis_label_fs
        )
        # plt.title("Predicted averaged spectrum map", fontsize=PLOT_LABEL_FS)
        plt.grid(False)



        # ax2d.axvline(0.684, color="white", linestyle="--", linewidth=1.2, alpha=0.95, zorder=10)
        # ax2d.annotate(
        #     r"$\omega_{\mathrm{finish}}$",
        #     xy=(0.684, 1),
        #     xytext=(14, -6),
        #     textcoords="offset points",
        #     fontsize=16,
        #     color="white",
        #     ha="left",
        #     va="top",
        #     zorder=11,
        # )

        # ax2d.axvline(0.675, color="white", linestyle="--", linewidth=1.2, alpha=0.95, zorder=10)
        # ax2d.annotate(
        #     r"$\omega_{\mathrm{start}}$",
        #     xy=(0.675, 1),
        #     xytext=(-14, -6),
        #     textcoords="offset points",
        #     fontsize=16,
        #     color="white",
        #     ha="right",
        #     va="top",
        #     zorder=11,
        # )

        ax2d.text(
            0.02,
            0.98,
            f"$a_0$ = {a0_ref:.1f}\n$\\tau$ = {(dur_ref / np.pi):.1f}$\\pi$",
            transform=ax2d.transAxes,
            ha="left",
            va="top",
            fontsize=vis_text_fs,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
            clip_on=False,
        )
        ax2d.set_ylim(0, np.nanmax(y_pos))
        plt.subplots_adjust(right=0.82, bottom=0.12)
        out_file_2d = "predicted_spectrum_theta_frequency_map.png"
        plt.savefig(out_file_2d, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig2d)
        print(f"2D карта предсказаний сохранена в {out_file_2d}")

        # --- Преобразование Лоренца для предсказанного спектра ---
        gamma0 = 100.0
        beta0 = np.sqrt(1.0 - 1.0/gamma0**2)
    
        grid_chi, grid_theta = np.meshgrid(freq_axis_2d, theta_axis)
    
        cos_theta_prime = np.cos(grid_theta)
        cos_theta_lab = (cos_theta_prime - beta0) / (1.0 - beta0 * cos_theta_prime)
        cos_theta_lab = np.clip(cos_theta_lab, -1.0, 1.0)
        theta_lab = np.arccos(cos_theta_lab)
    
        chi_lab = grid_chi * (1.0 - beta0 * cos_theta_prime) * (1.0 + beta0) / 4.0
        jacobian_full = 4.0 * gamma0**2 * (1.0 - beta0 * cos_theta_prime) / (1.0 + beta0)
        intensity_map_lab = intensity_map * jacobian_full
        y_pos_lab = gamma0 * (np.pi - theta_lab)
    
        fig2d_lab = plt.figure(figsize=(12, 7))
        zmax_pred_lab = float(np.nanmax(intensity_map_lab))
        mesh_lab = plt.pcolormesh(chi_lab, y_pos_lab, intensity_map_lab, shading="gouraud", cmap="viridis", vmin=0.0, vmax=zmax_pred_lab)
    
        plt.xlabel(r"$y_{\text{lab}}$", fontsize=vis_label_fs)
        plt.ylabel(r"$\gamma_0(\pi - \theta_{\text{lab}})$", fontsize=vis_label_fs)
    
        cbar_lab = plt.colorbar(mesh_lab, extend="neither")
        style_colorbar(
            cbar_lab,
            r"Predicted $\frac{d^2N}{d\omega_{\text{lab}} d\Omega_{\text{lab}}}$ Envelope",
            zmax_pred_lab,
            tick_fs=vis_tick_fs,
            label_fs=vis_cbar_label_fs,
        )
    
        ax2d_lab = plt.gca()
        ax2d_lab.set_facecolor(plt.cm.viridis(0))
        ax2d_lab.tick_params(axis="both", labelsize=vis_tick_fs)
        ticks = ax2d_lab.get_yticks()
        labels = [f"{t / gamma0:.3f}" for t in ticks]
        ax2d_lab.set_yticks(ticks)
        ax2d_lab.set_yticklabels(labels, fontsize=vis_tick_fs)
    
        ax2d_lab.set_ylim(0, np.nanmax(y_pos_lab))
        plt.xlim(0.01, 1.5)
        # plt.title(rf"Predicted Compton Spectrum in Lab Frame ($\gamma_0 = {gamma0}$)", fontsize=PLOT_LABEL_FS, pad=15)
    
        ax2d_lab.axhline(0.07, color="red", linestyle="--", linewidth=1, alpha=0.95, zorder=10)
        ax2d_lab.annotate(
            r"$\theta_c=0.0007$",
            xy=(0.6, 0.07),
            xytext=(-8, 6),
            textcoords="offset points",
            fontsize=vis_tick_fs,
            color="red",
            ha="right",
            va="bottom",
            zorder=11,
        )
    
        ax2d_lab.text(
            0.02,
            0.98,
            f"$a_0$ = {a0_ref:.1f}\n$\\tau$ = {(dur_ref / np.pi):.1f}$\\pi$",
            transform=ax2d_lab.transAxes,
            ha="left",
            va="top",
            fontsize=vis_text_fs,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
            clip_on=False,
        )
    
        plt.subplots_adjust(right=0.82, bottom=0.12)
        out_file_2d_lab = "predicted_spectrum_theta_frequency_map_lab.png"
        plt.savefig(out_file_2d_lab, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig2d_lab)
        print(f"2D карта предсказаний (Lab) сохранена в {out_file_2d_lab}")

        # --- 2D карта по аналитике (как в averaging_origin.ipynb) ---
        a0_eff = float(a0_ref) / np.sqrt(2.0)
        calc_analytic = D2ECalculator_angles(a0_eff, float(dur_ref), float(gamma_ref))
        grid_chi, grid_theta = np.meshgrid(
            freq_axis_2d.astype(np.float64),
            theta_axis.astype(np.float64),
            indexing="ij",
        )
        k_an = calc_analytic.k(grid_theta)
        b_an = calc_analytic.b(grid_theta)
        chi_bar = 1.0 / k_an
        xi_an = 1.0 / (k_an + float(a0_ref) ** 2 * b_an)
        y_mix = 0.5
        chi0_an = chi_bar - (chi_bar - xi_an) * y_mix
        pm0 = float(gamma_ref) + np.sqrt(float(gamma_ref) ** 2 - 1.0)
        d2E_an = calc_analytic.d2E(grid_chi, grid_theta, chi0_an)
        d2N_an = d2E_an / grid_chi / pm0**2
        d2N_an = np.nan_to_num(d2N_an, nan=0.0, posinf=0.0, neginf=0.0)

        # Двойной интеграл ∫∫ I(ω,θ) · 2π sin(θ) dω dθ (как по азимуту усреднённый dΩ)(средние по val (a0, τ))
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        theta_1d = np.asarray(theta_axis, dtype=np.float64)
        jac = 2.0 * np.pi * np.sin(theta_1d)[:, np.newaxis]
        eps_i = 1e-30

        # Ref-карта (a0=2, τ=50π): ∫∫ (d²N/dωdΩ)·2π sin θ dω dθ в полосе по осям графика
        # x = ω/(4γ²ω_L) ∈ [0.5, 1.25], y = γ(π−θ) ∈ [0, 0.2]
        f_reg_lo, f_reg_hi = 0.5, 1.25
        y_reg_hi = 0.2
        th_reg_lo = float(np.pi - y_reg_hi / float(gamma_ref))
        th_reg_hi = float(np.pi)
        fa_r, fb_r = min(f_reg_lo, f_reg_hi), max(f_reg_lo, f_reg_hi)
        ta_r, tb_r = min(th_reg_lo, th_reg_hi), max(th_reg_lo, th_reg_hi)
        f2_ref = np.asarray(freq_axis_2d, dtype=np.float64)
        freq_1d_ref = (
            f2_ref.ravel()
            if f2_ref.ndim == 1
            else f2_ref[0, :].ravel()
        )
        m_f_ref = (freq_1d_ref >= fa_r) & (freq_1d_ref <= fb_r)
        m_t_ref = (theta_1d >= ta_r) & (theta_1d <= tb_r)
        if np.any(m_f_ref) and np.any(m_t_ref):
            fq_r = freq_1d_ref[m_f_ref]
            th_r = theta_1d[m_t_ref]
            it_r = np.flatnonzero(m_t_ref)
            if_r = np.flatnonzero(m_f_ref)
            Zp_r = np.asarray(intensity_map, dtype=np.float64)[np.ix_(it_r, if_r)]
            Za_r = np.asarray(d2N_an.T, dtype=np.float64)[np.ix_(it_r, if_r)]
            jac_r = 2.0 * np.pi * np.sin(th_r)[:, np.newaxis]
            int_ref_strip_pred = float(_trapz(_trapz(Zp_r * jac_r, fq_r, axis=1), th_r))
            int_ref_strip_an = float(_trapz(_trapz(Za_r * jac_r, fq_r, axis=1), th_r))
            err_ref_strip_pct = (
                100.0 * abs(int_ref_strip_pred - int_ref_strip_an) / (abs(int_ref_strip_an) + eps_i)
                if np.isfinite(int_ref_strip_pred) and np.isfinite(int_ref_strip_an)
                else float("nan")
            )
            print(
                "\n--- Ref 2D (a0=2, τ=50π): ∫∫ (d²N/dωdΩ)·2π sin θ dω dθ ---\n"
                f"  Область: x∈[{f_reg_lo}, {f_reg_hi}], y=γ(π−θ)∈[0, {y_reg_hi}] "
                f"(θ∈[{ta_r:.6f}, {tb_r:.6f}] rad)\n"
                f"  Предсказание (intensity_map): {int_ref_strip_pred:.6e}\n"
                f"  Аналитика (d2N_an):           {int_ref_strip_an:.6e}\n"
                f"  Относительная ошибка:        {err_ref_strip_pct:.4f} %"
            )
        else:
            print(
                "\n--- Ref 2D: полоса x∈[0.675, 0.684], y∈[0, 0.2] не пересекается "
                "с узлами сетки (интеграл не посчитан)."
            )

        n_val_total = len(val_data)
        n_val_m = n_val_total if N_VAL_2D_METRICS_MAX is None else min(n_val_total, int(N_VAL_2D_METRICS_MAX))
        # В 2D-метрики попадают только сэмплы с a0 >= порога И tau >= порога (остальные отбрасываются по фильтру)
        val_2d_req_a0_ge = 1.0
        val_2d_req_tau_ge = 50.0 * np.pi
        int_preds, int_ans = [], []
        int_strip_preds, int_strip_ans = [], []
        rel_int_samples, norm_diffs, rel_norm_samples = [], [], []
        rel_int_strip_samples = []
        nn_2d_times, an_2d_times, quad_2d_times = [], [], []
        arr_phi_2d = np.array([0.0], dtype=np.float64)
        n_val_2d_quad_skipped = 0
        # Сравнение времени 2D-карт — все методы на CPU
        device_2d_cpu = torch.device("cpu")
        model_2d_cpu = DirectComptonModel()
        model_2d_cpu.load_state_dict(model.state_dict())
        model_2d_cpu.to(device_2d_cpu)
        model_2d_cpu.eval()
        normalizer_2d_cpu = DataNormalizer()
        normalizer_2d_cpu.p_mean = normalizer.p_mean.cpu()
        normalizer_2d_cpu.p_std = normalizer.p_std.cpu()
        normalizer_2d_cpu.y_mean = normalizer.y_mean.cpu()
        normalizer_2d_cpu.y_std = normalizer.y_std.cpu()
        # Полоса y_lab = γ_0(π−θ_lab) ∈ [0, 0.07], ω — весь диапазон сетки
        gamma0_lab = 100.0
        beta0_lab = np.sqrt(1.0 - 1.0/gamma0_lab**2)
        y_strip_hi = 0.07
    
        cos_theta_prime_1d = np.cos(theta_1d)
        cos_theta_lab_1d = (cos_theta_prime_1d - beta0_lab) / (1.0 - beta0_lab * cos_theta_prime_1d)
        cos_theta_lab_1d = np.clip(cos_theta_lab_1d, -1.0, 1.0)
        theta_lab_1d = np.arccos(cos_theta_lab_1d)
    
        y_pos_lab_1d = gamma0_lab * (np.pi - theta_lab_1d)
        m_t_strip = y_pos_lab_1d <= y_strip_hi
        it_strip = np.flatnonzero(m_t_strip)
        n_val_2d_skipped = 0
        n_val_2d_filter_skipped = 0
        n_val_2d_maps_saved = 0
        os.makedirs(VAL_2D_METRICS_MAPS_DIR, exist_ok=True)
        n_save_slots = min(int(N_VAL_2D_MAPS_TO_SAVE), n_val_m) if n_val_m > 0 else 0
        val_2d_map_slot_save = set(random.sample(range(n_val_m), n_save_slots)) if n_save_slots > 0 else set()
        # узкий фильтр (a0≥1 и τ≥50π) — больше попыток случайного draw
        max_2d_draw_attempts = max(500_000, 20_000 * max(n_val_m, 1))
        attempts_2d = 0
        collected_2d = 0
        while collected_2d < n_val_m and attempts_2d < max_2d_draw_attempts:
            attempts_2d += 1
            ri = random.randrange(n_val_total)
            p_i, _, freqs_i = val_data[ri]
            a0_i = float(p_i[0].item())
            dur_i = float(p_i[1].item())
            accept_2d_filter = (a0_i >= val_2d_req_a0_ge) and (dur_i >= val_2d_req_tau_ge)
            if not accept_2d_filter:
                n_val_2d_filter_skipped += 1
                continue
            freq_i = np.asarray(freqs_i.numpy().squeeze(), dtype=np.float64)
            p_grid_i = np.stack(
                [
                    np.full_like(theta_axis, a0_i, dtype=np.float32),
                    np.full_like(theta_axis, dur_i, dtype=np.float32),
                    theta_axis,
                ],
                axis=1,
            )
            dt_nn_2d = time.perf_counter()
            with torch.no_grad():
                p_grid_t_i = torch.from_numpy(p_grid_i).to(device_2d_cpu)
                p_norm_i = normalizer_2d_cpu.norm_params(p_grid_t_i)
                y_norm_i = model_2d_cpu(p_norm_i)
                y_log_i = normalizer_2d_cpu.unnorm_spec(y_norm_i).squeeze(1)
                y_lin_i = torch.clamp(10 ** y_log_i - 1e-9, min=0.0).numpy()
            dt_nn_2d = time.perf_counter() - dt_nn_2d
            I_pred_i = np.asarray(y_lin_i, dtype=np.float64)
            if not np.isfinite(I_pred_i).all():
                n_val_2d_skipped += 1
                continue

            dt_an_2d = time.perf_counter()
            a0_eff_i = a0_i / np.sqrt(2.0)
            calc_i = D2ECalculator_angles(a0_eff_i, float(dur_i), float(gamma_ref))
            g_chi, g_th = np.meshgrid(freq_i.astype(np.float64), theta_1d, indexing="ij")
            k_i = calc_i.k(g_th)
            b_i = calc_i.b(g_th)
            chi_bar_i = 1.0 / k_i
            xi_i = 1.0 / (k_i + float(a0_i) ** 2 * b_i)
            chi0_i = chi_bar_i - (chi_bar_i - xi_i) * 0.5
            pm0_i = float(gamma_ref) + np.sqrt(float(gamma_ref) ** 2 - 1.0)
            d2E_i = calc_i.d2E(g_chi, g_th, chi0_i)
            if not np.isfinite(d2E_i).all():
                n_val_2d_skipped += 1
                continue
            d2N_i = d2E_i / g_chi / pm0_i**2
            if not np.isfinite(d2N_i).all():
                n_val_2d_skipped += 1
                continue
            I_an_i = np.asarray(d2N_i.T, dtype=np.float64)
            dt_an_2d = time.perf_counter() - dt_an_2d

            dt_quad_2d = time.perf_counter()
            I_quad_i = None
            try:
                spectrum_sqinrp_i = init_spectrum_sqinrp(root, a0_i, dur_i, float(gamma_ref))
                spec_quad_i, _ = spectrum_sqinrp_i.get_spectrum(
                    arr_phi_2d,
                    theta_1d.astype(np.float64),
                    freq_i.astype(np.float64),
                )
                I_quad_i = np.asarray(spec_quad_i[0], dtype=np.float64) / freq_i[np.newaxis, :] / pm0_i**2
            except Exception:
                n_val_2d_quad_skipped += 1
            dt_quad_2d = time.perf_counter() - dt_quad_2d
            if I_quad_i is not None and not np.isfinite(I_quad_i).all():
                n_val_2d_quad_skipped += 1
                I_quad_i = None

            I_pw = I_pred_i * jac
            I_aw = I_an_i * jac
            if not (np.isfinite(I_pw).all() and np.isfinite(I_aw).all()):
                n_val_2d_skipped += 1
                continue
            ip = float(_trapz(_trapz(I_pw, freq_i, axis=1), theta_1d))
            ia = float(_trapz(_trapz(I_aw, freq_i, axis=1), theta_1d))
            if not (np.isfinite(ip) and np.isfinite(ia)):
                n_val_2d_skipped += 1
                continue
            diff_i = I_pred_i - I_an_i
            nd = float(np.linalg.norm(diff_i, ord="fro"))
            na = float(np.linalg.norm(I_an_i, ord="fro"))
            if not (np.isfinite(nd) and np.isfinite(na) and na > 0.0):
                n_val_2d_skipped += 1
                continue
            if collected_2d in val_2d_map_slot_save:
                pred_png = os.path.join(VAL_2D_METRICS_MAPS_DIR, f"val_{collected_2d:04d}_predicted.png")
                an_png = os.path.join(VAL_2D_METRICS_MAPS_DIR, f"val_{collected_2d:04d}_analytical.png")
                save_val_2d_spectrum_map_png(
                    pred_png,
                    freq_i,
                    y_pos,
                    I_pred_i,
                    float(gamma_ref),
                    r"Predicted $\frac{d^2N}{d\omega d\Omega}$",
                    a0_i,
                    dur_i,
                    tick_fs=vis_tick_fs,
                    label_fs=vis_label_fs,
                    cbar_label_fs=vis_cbar_label_fs,
                    text_fs=vis_text_fs,
                )
                save_val_2d_spectrum_map_png(
                    an_png,
                    freq_i,
                    y_pos,
                    I_an_i,
                    float(gamma_ref),
                    r"Analytical $\frac{d^2N}{d\omega d\Omega}$",
                    a0_i,
                    dur_i,
                    tick_fs=vis_tick_fs,
                    label_fs=vis_label_fs,
                    cbar_label_fs=vis_cbar_label_fs,
                    text_fs=vis_text_fs,
                )
                if I_quad_i is not None:
                    quad_png = os.path.join(VAL_2D_METRICS_MAPS_DIR, f"val_{collected_2d:04d}_quad_interp.png")
                    save_val_2d_spectrum_map_png(
                        quad_png,
                        freq_i,
                        y_pos,
                        I_quad_i,
                        float(gamma_ref),
                        r"quad\_interp $\frac{d^2N}{d\omega d\Omega}$",
                        a0_i,
                        dur_i,
                        tick_fs=vis_tick_fs,
                        label_fs=vis_label_fs,
                        cbar_label_fs=vis_cbar_label_fs,
                        text_fs=vis_text_fs,
                    )
                n_val_2d_maps_saved += 1
            int_preds.append(ip)
            int_ans.append(ia)
            rel_int_samples.append(100.0 * abs(ip - ia) / (abs(ia) + eps_i))
            if it_strip.size > 0:
                th_s = theta_1d[it_strip]
                I_ps = I_pred_i[it_strip, :]
                I_as = I_an_i[it_strip, :]
                jac_s = 2.0 * np.pi * np.sin(th_s)[:, np.newaxis]
                ip_strip = float(_trapz(_trapz(I_ps * jac_s, freq_i, axis=1), th_s))
                ia_strip = float(_trapz(_trapz(I_as * jac_s, freq_i, axis=1), th_s))
            else:
                ip_strip = float("nan")
                ia_strip = float("nan")
            int_strip_preds.append(ip_strip)
            int_strip_ans.append(ia_strip)
            rel_int_strip_samples.append(
                100.0 * abs(ip_strip - ia_strip) / (abs(ia_strip) + eps_i)
                if np.isfinite(ip_strip) and np.isfinite(ia_strip)
                else float("nan")
            )
            norm_diffs.append(nd)
            rel_norm_samples.append(100.0 * nd / (na + eps_i))
            nn_2d_times.append(dt_nn_2d)
            an_2d_times.append(dt_an_2d)
            quad_2d_times.append(dt_quad_2d)
            collected_2d += 1

        if collected_2d < n_val_m:
            lim_note = (
                f" Достигнут лимит {max_2d_draw_attempts} попыток отбора."
                if attempts_2d >= max_2d_draw_attempts
                else ""
            )
            print(
                f"\n  ВНИМАНИЕ: набрано только {collected_2d} из {n_val_m} сэмплов для 2D-метрик.{lim_note}\n"
                f"    (val={n_val_total}, в метрики берутся только сэмплы с одновременно a0>={val_2d_req_a0_ge} и "
                f"τ>={val_2d_req_tau_ge/np.pi:.0f}π; отброшено по фильтру: {n_val_2d_filter_skipped}, по числам inf/nan: {n_val_2d_skipped})."
            )

        int_preds = np.asarray(int_preds, dtype=np.float64)
        int_ans = np.asarray(int_ans, dtype=np.float64)
        int_strip_preds = np.asarray(int_strip_preds, dtype=np.float64)
        int_strip_ans = np.asarray(int_strip_ans, dtype=np.float64)
        rel_int_samples = np.asarray(rel_int_samples, dtype=np.float64)
        rel_int_strip_samples = np.asarray(rel_int_strip_samples, dtype=np.float64)
        norm_diffs = np.asarray(norm_diffs, dtype=np.float64)
        rel_norm_samples = np.asarray(rel_norm_samples, dtype=np.float64)
        n_val_2d_used = int_preds.size
        if n_val_2d_used > 0:
            rel_int_mean_vs_mean = 100.0 * abs(np.mean(int_preds) - np.mean(int_ans)) / (abs(np.mean(int_ans)) + eps_i)
            rel_int_strip_mean_vs_mean = 100.0 * abs(
                np.mean(int_strip_preds) - np.mean(int_strip_ans)
            ) / (abs(np.mean(int_strip_ans)) + eps_i)
        else:
            rel_int_mean_vs_mean = float("nan")
            rel_int_strip_mean_vs_mean = float("nan")

        print("\n--- 2D метрики (случайные сэмплы из val; сетка θ как на картах) ---")
        if n_val_m == n_val_total:
            print(f"  Целевое число сэмплов для усреднения: {n_val_m} (всего в val: {n_val_total}).")
        else:
            print(
                f"  Целевое число сэмплов для усреднения: {n_val_m} "
                f"(всего в val: {n_val_total}, лимит N_VAL_2D_METRICS_MAX={N_VAL_2D_METRICS_MAX})."
            )
        print(
            f"  Отбор в 2D-метрики: только сэмплы с одновременно a0>={val_2d_req_a0_ge} и "
            f"τ>={val_2d_req_tau_ge/np.pi:.0f}π (остальные отброшено по фильтру: {n_val_2d_filter_skipped})"
        )
        if n_val_2d_maps_saved:
            print(
                f"  Сохранены случайные 2D-карты (не более {N_VAL_2D_MAPS_TO_SAVE} пар из {n_val_m} в метриках): "
                f"{n_val_2d_maps_saved} пар PNG → {VAL_2D_METRICS_MAPS_DIR}"
            )
        if n_val_2d_skipped:
            print(f"  Отброшено по числам (inf/nan в аналитике/предсказании/интеграле): {n_val_2d_skipped}")
        if n_val_2d_used == 0:
            print("  Нет ни одного конечного сэмпла для 2D-метрик (все пропущены).")
        else:
            print(f"  Учтено конечных сэмплов: {n_val_2d_used}" + (f" из {n_val_m} целевых" if n_val_2d_used != n_val_m else ""))
            print(f"  Средний интеграл ∫∫ I·2π sin θ dω dθ (нейросеть): {np.mean(int_preds):.6e}  (std {np.std(int_preds):.6e})")
            print(f"  Средний интеграл (аналитика):                  {np.mean(int_ans):.6e}  (std {np.std(int_ans):.6e})")
            print(f"  Средняя относ. ошибка интеграла по сэмплам:     {np.mean(rel_int_samples):.4f} %  (std {np.std(rel_int_samples):.4f} %)")
            print(f"  Относ. ошибка средних интегралов:               {rel_int_mean_vs_mean:.4f} %")
            print(f"  Средняя ||I_pred - I_an||_F:                     {np.mean(norm_diffs):.6e}  (std {np.std(norm_diffs):.6e})")
            print(f"  Средняя относ. норма Фробениуса по сэмплам:     {np.mean(rel_norm_samples):.4f} %  (std {np.std(rel_norm_samples):.4f} %)")
            print(
                f"\n  Полоса y=γ_0(π−θ_lab)∈[0, {y_strip_hi}] (Lab frame), ω — весь диапазон: ∫∫ I·2π sin θ dω dθ"
            )
            print(
                f"  Средний интеграл (нейросеть): {np.mean(int_strip_preds):.6e}  (std {np.std(int_strip_preds):.6e})"
            )
            print(
                f"  Средний интеграл (аналитика): {np.mean(int_strip_ans):.6e}  (std {np.std(int_strip_ans):.6e})"
            )
            print(
                f"  Средняя относ. ошибка интеграла по сэмплам: {np.nanmean(rel_int_strip_samples):.4f} %  "
                f"(std {np.nanstd(rel_int_strip_samples):.4f} %)"
            )
            print(f"  Относ. ошибка средних интегралов:          {rel_int_strip_mean_vs_mean:.4f} %")
            print("\n  Сравнение времени 2D-карт (все методы на CPU):")
            print(
                f"  Среднее время расчёта 2D-карты на 1 сэмпл (нейросеть, CPU): "
                f"{np.mean(nn_2d_times) * 1000:.3f} мс  (std {np.std(nn_2d_times) * 1000:.3f} мс, n={len(nn_2d_times)})"
            )
            print(
                f"  Среднее время расчёта 2D-карты на 1 сэмпл (D2ECalculator_angles, CPU): "
                f"{np.mean(an_2d_times) * 1000:.3f} мс  (std {np.std(an_2d_times) * 1000:.3f} мс, n={len(an_2d_times)})"
            )
            if quad_2d_times:
                print(
                    f"  Среднее время расчёта 2D-карты на 1 сэмпл (quad_interp / snlcs, CPU): "
                    f"{np.mean(quad_2d_times) * 1000:.3f} мс  (std {np.std(quad_2d_times) * 1000:.3f} мс, n={len(quad_2d_times)})"
                )
            if n_val_2d_quad_skipped:
                print(
                    f"  quad_interp: не удалось получить конечную 2D-карту для "
                    f"{n_val_2d_quad_skipped} из {len(quad_2d_times)} попыток"
                )

        fig2d_an = plt.figure(figsize=(12, 7))
        zmax_an = float(np.nanmax(d2N_an.T))
        mesh_an = plt.pcolormesh(
            freq_axis_2d,
            y_pos,
            d2N_an.T,
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=zmax_an,
        )
        cbar_an = plt.colorbar(mesh_an, extend="neither")
        style_colorbar(
            cbar_an,
            r"Analytical $\frac{d^2N}{d\omega d\Omega}$ Envelope",
            zmax_an,
            tick_decimals=None,
            scale_exp=0,
            tick_fs=vis_tick_fs,
            label_fs=vis_cbar_label_fs,
        )
        ax2d_an = plt.gca()
        style_spectrum_2d_map_axes(
            ax2d_an, float(gamma_ref), xlim=(0.01, 1.5), tick_fs=vis_tick_fs, label_fs=vis_label_fs
        )
        # plt.title("Analytical spectrum map (averaging_origin recipe)", fontsize=PLOT_LABEL_FS)


        # ax2d_an.axvline(0.684, color="white", linestyle="--", linewidth=1.2, alpha=0.95, zorder=10)
        # ax2d_an.annotate(
        #     r"$\omega_{\mathrm{finish}}$",
        #     xy=(0.684, 1),
        #     xytext=(14, -6),
        #     textcoords="offset points",
        #     fontsize=16,
        #     color="white",
        #     ha="left",
        #     va="top",
        #     zorder=11,
        # )

        # ax2d_an.axvline(0.675, color="white", linestyle="--", linewidth=1.2, alpha=0.95, zorder=10)
        # ax2d_an.annotate(
        #     r"$\omega_{\mathrm{start}}$",
        #     xy=(0.675, 1),
        #     xytext=(-14, -6),
        #     textcoords="offset points",
        #     fontsize=16,
        #     color="white",
        #     ha="right",
        #     va="top",
        #     zorder=11,
        # )

        plt.grid(False)
        ax2d_an.text(
            0.02,
            0.98,
            f"$a_0$ = {a0_ref:.1f}\n$\\tau$ = {(dur_ref / np.pi):.1f}$\\pi$",
            transform=ax2d_an.transAxes,
            ha="left",
            va="top",
            fontsize=vis_text_fs,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
            clip_on=False,
        )
        ax2d_an.set_ylim(0, np.nanmax(y_pos))
        plt.subplots_adjust(right=0.82, bottom=0.12)
        out_file_2d_an = "analytical_spectrum_theta_frequency_map.png"
        plt.savefig(out_file_2d_an, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig2d_an)
        print(f"2D карта (аналитика) сохранена в {out_file_2d_an}")

        # --- Преобразование Лоренца для аналитического спектра ---
        d2N_an_lab = d2N_an.T * jacobian_full
    
        fig2d_an_lab = plt.figure(figsize=(12, 7))
        zmax_an_lab = float(np.nanmax(d2N_an_lab))
        mesh_an_lab = plt.pcolormesh(chi_lab, y_pos_lab, d2N_an_lab, shading="gouraud", cmap="viridis", vmin=0.0, vmax=zmax_an_lab)
    
        plt.xlabel(r"$y_{\text{lab}}$", fontsize=vis_label_fs)
        plt.ylabel(r"$\gamma_0(\pi - \theta_{\text{lab}})$", fontsize=vis_label_fs)
    
        cbar_an_lab = plt.colorbar(mesh_an_lab, extend="neither")
        style_colorbar(
            cbar_an_lab,
            r"Analytical $\frac{d^2N}{d\omega_{\text{lab}} d\Omega_{\text{lab}}}$ Envelope",
            zmax_an_lab,
            tick_fs=vis_tick_fs,
            label_fs=vis_cbar_label_fs,
        )
    
        ax2d_an_lab = plt.gca()
        ax2d_an_lab.set_facecolor(plt.cm.viridis(0))
        ax2d_an_lab.tick_params(axis="both", labelsize=vis_tick_fs)
        ticks = ax2d_an_lab.get_yticks()
        labels = [f"{t / gamma0:.3f}" for t in ticks]
        ax2d_an_lab.set_yticks(ticks)
        ax2d_an_lab.set_yticklabels(labels, fontsize=vis_tick_fs)
    
        ax2d_an_lab.set_ylim(0, np.nanmax(y_pos_lab))
        plt.xlim(0.01, 1.5)
        # plt.title(rf"Analytical Compton Spectrum in Lab Frame ($\gamma_0 = {gamma0}$)", fontsize=PLOT_LABEL_FS, pad=15)
    
        ax2d_an_lab.axhline(0.07, color="red", linestyle="--", linewidth=1, alpha=0.95, zorder=10)
        ax2d_an_lab.annotate(
            r"$\theta_c=0.0007$",
            xy=(0.6, 0.07),
            xytext=(-8, 6),
            textcoords="offset points",
            fontsize=vis_tick_fs,
            color="red",
            ha="right",
            va="bottom",
            zorder=11,
        )
    
        ax2d_an_lab.text(
            0.02,
            0.98,
            f"$a_0$ = {a0_ref:.1f}\n$\\tau$ = {(dur_ref / np.pi):.1f}$\\pi$",
            transform=ax2d_an_lab.transAxes,
            ha="left",
            va="top",
            fontsize=vis_text_fs,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
            clip_on=False,
        )
    
        plt.subplots_adjust(right=0.82, bottom=0.12)
        out_file_2d_an_lab = "analytical_spectrum_theta_frequency_map_lab.png"
        plt.savefig(out_file_2d_an_lab, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig2d_an_lab)
        print(f"2D карта (аналитика Lab) сохранена в {out_file_2d_an_lab}")
