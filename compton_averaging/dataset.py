"""Генерация и загрузка датасета Compton."""
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.calc_spectrum import (
    D2ECalculator_angles,
    calculate_N_photons_with_freqs,
    calculate_trajectory,
)


class ComptonDataset(Dataset):
    def __init__(self, num_samples=500, timesteps=2500, cache_dir=None, seed=42):
        self.num_samples = num_samples
        self.timesteps = timesteps
        self.seed = seed
        self.gamma = 1.0
        self.target_freqs = np.linspace(0.01, 2.0, self.timesteps, dtype=np.float32)
        self.a0_vals = np.linspace(0.01, 2, 1800)
        self.duration_vals = np.arange(20, 201, 2) * np.pi

        self.cache_dir = cache_dir
        self._cache_ready = False
        self._cache_bundle = None
        self._cache_p = None
        self._cache_y = None
        self._cache_freqs = None
        self.data = None  # only when cache_dir is None

        cache_format = "npz_bundle_v1"
        sampling_version = "balanced_v1"
        bundle_name = "dataset.npz"

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            meta_path = os.path.join(self.cache_dir, "meta.pkl")
            bundle_path = os.path.join(self.cache_dir, bundle_name)

            if os.path.isfile(meta_path) and os.path.isfile(bundle_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                if (
                    meta.get("cache_format") == cache_format
                    and meta.get("sampling_version") == sampling_version
                    and meta.get("timesteps") == timesteps
                    and meta.get("seed") == seed
                ):
                    bundle = np.load(bundle_path, mmap_mode="r")
                    p_arr = bundle["p"]
                    y_arr = bundle["y"]
                    meta_n = int(meta.get("num_samples", -1))
                    ok = (
                        p_arr.ndim == 2
                        and p_arr.shape[1] == 3
                        and y_arr.ndim == 2
                        and y_arr.shape[1] == timesteps
                        and meta_n == p_arr.shape[0] == y_arr.shape[0]
                    )
                    if ok:
                        self._cache_bundle = bundle
                        self._cache_p = p_arr
                        self._cache_y = y_arr
                        self._cache_freqs = bundle["freqs"]
                        self._cache_ready = True
                        self.num_samples = meta_n
                        print(f"Загрузка датасета из кеша (один файл): {bundle_path}")
                    else:
                        bundle.close()
                        print("Найден кеш, но формат/размер не совпал — пересоздаём.")

        if self.cache_dir is None or not self._cache_ready:
            random.seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)

            if self.cache_dir is None:
                self.data = []
            else:
                all_p = np.empty((num_samples, 3), dtype=np.float32)
                all_y = np.empty((num_samples, timesteps), dtype=np.float32)
                shared_freqs = self.target_freqs.copy()

            print("Генерация датасета..." if self.cache_dir is None else "Генерация датасета и запись в кеш...")

            # Сбалансированное расписание пар (a0, dur): все комбинации индексов, затем перемешивание.
            na = len(self.a0_vals)
            nd = len(self.duration_vals)
            ia, idur = np.meshgrid(np.arange(na), np.arange(nd), indexing="ij")
            pairs = np.stack([ia.ravel(), idur.ravel()], axis=1)
            rng.shuffle(pairs)
            reps = int(np.ceil(num_samples / len(pairs)))
            pair_schedule = np.tile(pairs, (reps, 1))
            rng.shuffle(pair_schedule)
            pair_schedule = pair_schedule[:num_samples]

            for i in range(num_samples):
                a0 = float(self.a0_vals[pair_schedule[i, 0]])
                dur = float(self.duration_vals[pair_schedule[i, 1]])

                # Для каждой пары (a0, dur): theta равномерно в допустимом диапазоне.
                theta_lo = np.pi/2 # np.pi - a0 / self.gamma
                theta_hi = np.pi
                theta = float(rng.random() * (theta_hi - theta_lo) + theta_lo)
                x_params = np.array([a0, dur, theta], dtype=np.float32)

                if a0**2*dur >= 5:
                    arr_chi = self.target_freqs.copy()
                    pm = self.gamma + np.sqrt(self.gamma**2 - 1)
                    arr_theta = np.array([theta], dtype=np.float64)
                    grid_chi, grid_theta = np.meshgrid(arr_chi, arr_theta, indexing="ij")

                    calc = D2ECalculator_angles(a0=a0/np.sqrt(2), T=dur, gamma=self.gamma)
                    k = calc.k(grid_theta)
                    b = calc.b(grid_theta)
                    chi_bar = 1.0 / k
                    xi = 1.0 / (k + a0**2*b)
                    y = 0.5
                    chi0 = chi_bar-(chi_bar - xi) * y

                    d2E = calc.d2E(grid_chi, grid_theta, chi0)
                    y_target = (d2E / grid_chi / pm**2).T[-1]
                    freqs = np.array(arr_chi, dtype=np.float32)
                elif a0 >= 0.1:
                    t = np.linspace(-100 * 2 * np.pi, 100 * 2 * np.pi, timesteps)
                    state = calculate_trajectory([0.0] * 6, t, a0=a0, duration=dur)
                    y_target, freqs_raw = calculate_N_photons_with_freqs(state, t, theta=theta)
                    y_target = np.interp(self.target_freqs, freqs_raw, y_target, left=0.0, right=0.0)
                    freqs = self.target_freqs.copy()
                else:
                    t = np.linspace(-50 * 2 * np.pi, 50 * 2 * np.pi, timesteps)
                    state = calculate_trajectory([0.0] * 6, t, a0=a0, duration=dur)
                    y_target, freqs_raw = calculate_N_photons_with_freqs(state, t, theta=theta)
                    y_target = np.interp(self.target_freqs, freqs_raw, y_target, left=0.0, right=0.0)
                    freqs = self.target_freqs.copy()

                y_target = np.asarray(y_target, dtype=np.float32)
                y_target = np.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
                y_target = np.clip(y_target, a_min=0.0, a_max=None)
                freqs = np.asarray(freqs, dtype=np.float32)
                y_target_log = np.log10(y_target + 1e-9).astype(np.float32)

                if self.cache_dir is None:
                    self.data.append((x_params, y_target_log, freqs))
                else:
                    all_p[i] = x_params
                    all_y[i] = y_target_log

                if (i + 1) % 100 == 0:
                    print(f"Сгенерировано {i+1}/{num_samples}")

            if self.cache_dir is not None:
                bundle_path = os.path.join(self.cache_dir, bundle_name)
                np.savez_compressed(bundle_path, p=all_p, y=all_y, freqs=shared_freqs)
                meta_path = os.path.join(self.cache_dir, "meta.pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(
                        {
                            "num_samples": num_samples,
                            "timesteps": timesteps,
                            "seed": seed,
                            "cache_format": cache_format,
                            "sampling_version": sampling_version,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                self._cache_bundle = np.load(bundle_path, mmap_mode="r")
                self._cache_p = self._cache_bundle["p"]
                self._cache_y = self._cache_bundle["y"]
                self._cache_freqs = self._cache_bundle["freqs"]
                self._cache_ready = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._cache_ready:
            p = np.asarray(self._cache_p[idx], dtype=np.float32)
            y = np.asarray(self._cache_y[idx], dtype=np.float32)
            ff = self._cache_freqs
            if ff.ndim == 1:
                f = np.asarray(ff, dtype=np.float32)
            else:
                f = np.asarray(ff[idx], dtype=np.float32)
        else:
            p, y, f = self.data[idx]
            p = np.asarray(p, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            f = np.asarray(f, dtype=np.float32)

        return (
            torch.from_numpy(p).float(),
            torch.from_numpy(y).unsqueeze(0).float(),
            torch.from_numpy(f).float(),
        )


def stratified_train_val_indices(p, train_fraction=0.8, seed=42, gamma=1.0, n_bins=5):
    """
    Стратифицированные индексы train/val по (a0, dur, theta_norm),
    где theta_norm = (theta - (pi - a0/gamma)) / (a0/gamma).
    """
    rng = np.random.default_rng(seed + 10007)
    p = np.asarray(p, dtype=np.float64)
    n = p.shape[0]
    a0, dur, theta = p[:, 0], p[:, 1], p[:, 2]

    theta_min = np.pi - a0 / gamma
    span = np.pi - theta_min
    theta_norm = np.divide(theta - theta_min, span, out=np.full(n, 0.5), where=span > 1e-12)

    a0_b = np.clip(((a0 - 0.01) / (2.0 - 0.01) * n_bins).astype(np.int64), 0, n_bins - 1)
    dur_lo = float(dur.min())
    dur_r = float(np.ptp(dur)) + 1e-12
    dur_b = np.clip(((dur - dur_lo) / dur_r * n_bins).astype(np.int64), 0, n_bins - 1)
    th_b = np.clip((theta_norm * n_bins).astype(np.int64), 0, n_bins - 1)

    strata = a0_b + n_bins * (dur_b + n_bins * th_b)
    train_parts, val_parts = [], []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) == 0:
            continue
        perm = rng.permutation(idx)
        if len(idx) == 1:
            if rng.random() < train_fraction:
                train_parts.append(perm)
            else:
                val_parts.append(perm)
            continue
        n_tr = int(np.floor(train_fraction * len(idx)))
        n_tr = int(np.clip(n_tr, 1, len(idx) - 1))
        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:])

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_indices = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    return train_indices.tolist(), val_indices.tolist()


def make_stratified_train_val_subsets(dataset, train_fraction=0.8, seed=42):
    if getattr(dataset, "_cache_ready", False) and dataset._cache_p is not None:
        p_all = np.asarray(dataset._cache_p)
    else:
        p_all = np.stack([dataset.data[i][0] for i in range(len(dataset))], axis=0)
    tr_idx, va_idx = stratified_train_val_indices(
        p_all, train_fraction=train_fraction, seed=seed, gamma=dataset.gamma
    )
    train_data = torch.utils.data.Subset(dataset, tr_idx)
    val_data = torch.utils.data.Subset(dataset, va_idx)
    return train_data, val_data, va_idx
