"""Benchmark NN vs D2ECalculator_angles 2D spectrum on CPU (compute only)."""
import os
import pickle
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from compton_averaging.config import model_file, normalizer_file
from compton_averaging.model import DirectComptonModel
from compton_averaging.normalizer import DataNormalizer
from snlcs.ebeam import eGaussBeam
from snlcs.external_fields import ExternalField
from src.calc_spectrum import D2ECalculator_angles



def main():
    # --- setup (not timed) ---
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    path_extf_params = os.path.join(BASE_DIR, "config/ExtFieldsParams.yaml")
    path_ebeam_params = os.path.join(BASE_DIR, "config/eBeamParams.yaml")

    extfields_params = ExternalField.from_yaml(path_extf_params)
    ebeam_params = eGaussBeam.from_yaml(path_ebeam_params)
    in_states = ebeam_params.initial_states
    N_particles = in_states.shape[0]

    a0_lab = extfields_params.a0
    dur_lab = extfields_params.T

    with open(normalizer_file(BASE_DIR), "rb") as f:
        normalizer = pickle.load(f)
    normalizer.to(device)

    model = DirectComptonModel(params_dim=3).to(device)
    checkpoint = torch.load(
        model_file(BASE_DIR),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

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
    freq_axis_2d = np.linspace(0.01, 2.0, 2500).astype(np.float32)

    nx = np.sin(THETA_LAB)
    ny = 0.0
    nz = np.cos(THETA_LAB)

    # precompute observation geometry (same for all particles)
    nx_obs_1d = np.sin(theta_lab_grid)
    ny_obs_1d = 0.0
    nz_obs_1d = np.cos(theta_lab_grid)
    row_idx = np.broadcast_to(np.arange(len(theta_lab_grid))[:, None], CHI_LAB.shape)
    dx = freq_axis_2d[1] - freq_axis_2d[0]


    def run_nn_spectrum():
        total = np.zeros_like(CHI_LAB)
        nn_forward = 0.0
        lorentz_interp = 0.0

        for i in range(N_particles):
            px, py, pz = in_states[i, 3], in_states[i, 4], in_states[i, 5]
            p_mag = np.sqrt(px**2 + py**2 + pz**2)
            gamma_i = np.sqrt(1 + p_mag**2)
            beta_vec = np.array([px, py, pz]) / gamma_i
            beta_i = np.linalg.norm(beta_vec)

            beta_dot_n_1d = (
                nx_obs_1d * beta_vec[0]
                + ny_obs_1d * beta_vec[1]
                + nz_obs_1d * beta_vec[2]
            )
            n_prime_z_1d = (
                nz_obs_1d
                - gamma_i * beta_vec[2]
                + (gamma_i**2 / (gamma_i + 1.0)) * beta_dot_n_1d * beta_vec[2]
            ) / (gamma_i * (1.0 - beta_dot_n_1d))
            cos_theta_prime_1d = np.clip(n_prime_z_1d, -1.0, 1.0)
            theta_prime_1d = np.arccos(cos_theta_prime_1d).astype(np.float32)

            p_grid = np.stack(
                [
                    np.full_like(theta_prime_1d, a0_lab),
                    np.full_like(theta_prime_1d, dur_lab),
                    theta_prime_1d,
                ],
                axis=1,
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                p_grid_t = torch.from_numpy(p_grid).to(device)
                p_grid_norm = normalizer.norm_params(p_grid_t)
                y_grid_norm = model(p_grid_norm)
                y_grid_log = normalizer.unnorm_spec(y_grid_norm).squeeze(1)
                y_grid_lin = torch.clamp(10 ** y_grid_log - 1e-9, min=0.0).cpu().numpy()
            nn_forward += time.perf_counter() - t0

            t1 = time.perf_counter()
            cos_theta_prime_2d = np.broadcast_to(cos_theta_prime_1d[:, None], CHI_LAB.shape)
            chi_prime = CHI_LAB * 4.0 / ((1.0 - beta_i * cos_theta_prime_2d) * (1.0 + beta_i))

            idx = (chi_prime - freq_axis_2d[0]) / dx
            idx_floor = np.floor(idx).astype(int)
            idx_ceil = idx_floor + 1
            weight_ceil = idx - idx_floor
            weight_floor = 1.0 - weight_ceil
            valid = (idx_floor >= 0) & (idx_ceil < len(freq_axis_2d))

            intensity_prime = np.zeros_like(chi_prime)
            intensity_prime[valid] = (
                y_grid_lin[row_idx[valid], idx_floor[valid]] * weight_floor[valid]
                + y_grid_lin[row_idx[valid], idx_ceil[valid]] * weight_ceil[valid]
            )

            jacobian_1d = 4.0 * gamma_i**2 * (1.0 - beta_i * cos_theta_prime_1d) / (1.0 + beta_i)
            jacobian_full = np.broadcast_to(jacobian_1d[:, None], CHI_LAB.shape)
            total += intensity_prime * jacobian_full
            lorentz_interp += time.perf_counter() - t1

        total /= N_particles
        return total, nn_forward, lorentz_interp


    def run_analytic_spectrum():
        total = np.zeros_like(CHI_LAB)
        d2e_time = 0.0

        for i in range(N_particles):
            px, py, pz = in_states[i, 3], in_states[i, 4], in_states[i, 5]
            p_mag = np.sqrt(px**2 + py**2 + pz**2)
            gamma_p = np.sqrt(1 + p_mag**2)

            cos_theta_eff = (px * nx + py * ny + pz * nz) / p_mag
            cos_theta_eff = np.clip(cos_theta_eff, -1.0, 1.0)
            theta_eff = np.arccos(cos_theta_eff)
            theta_input = np.pi - theta_eff

            calc_analytic = D2ECalculator_angles(a0_lab / np.sqrt(2), dur_lab, gamma_p)
            k = calc_analytic.k(theta_input)
            b = calc_analytic.b(theta_input)
            chi_bar = 1.0 / k
            xi = 1.0 / (k + (a0_lab / np.sqrt(2)) ** 2 * b)
            chi0 = chi_bar - (chi_bar - xi) * 0.5

            t0 = time.perf_counter()
            d2E_analytic = calc_analytic.d2E(CHI_LAB, theta_input, chi0)
            d2N_p = d2E_analytic / CHI_LAB
            d2e_time += time.perf_counter() - t0

            total += d2N_p

        total /= N_particles
        return total, d2e_time


    # warmup
    _ = run_nn_spectrum()[0]
    _ = run_analytic_spectrum()[0]

    n_repeat = 3
    nn_times, nn_fwd, nn_lorentz = [], [], []
    an_times, an_d2e = [], []

    for _ in range(n_repeat):
        _, f, l = run_nn_spectrum()
        nn_fwd.append(f)
        nn_lorentz.append(l)
        nn_times.append(f + l)

        _, d = run_analytic_spectrum()
        an_d2e.append(d)
        an_times.append(d)

    print("=== CPU benchmark (compute only, no imports/plotting) ===")
    print(f"Device: {device}, torch threads: {torch.get_num_threads()}")
    print(f"Particles: {N_particles}, grid: {theta_N} x {freq_N} = {theta_N * freq_N}")
    print(f"NN internal grid: {theta_N} angles x {len(freq_axis_2d)} freq bins")
    print(f"Repeats: {n_repeat}")
    print()
    print(f"NN total:           {np.mean(nn_times):.3f} ± {np.std(nn_times):.3f} s")
    print(f"  - forward pass:   {np.mean(nn_fwd):.3f} ± {np.std(nn_fwd):.3f} s")
    print(f"  - Lorentz+interp: {np.mean(nn_lorentz):.3f} ± {np.std(nn_lorentz):.3f} s")
    print(f"D2ECalculator_angles d2E: {np.mean(an_d2e):.3f} ± {np.std(an_d2e):.3f} s")
    print(f"Ratio (analytic / NN): {np.mean(an_d2e) / np.mean(nn_times):.2f}x")
    print(f"Per particle NN:       {np.mean(nn_times) / N_particles * 1000:.1f} ms")
    print(f"Per particle analytic: {np.mean(an_d2e) / N_particles * 1000:.1f} ms")


if __name__ == "__main__":
    main()
