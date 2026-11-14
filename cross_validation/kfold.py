"""Стратифицированный K-fold по тем же стратам, что train/val в compton_averaging.dataset."""
import numpy as np


def _compute_strata(p: np.ndarray, gamma: float = 1.0, n_bins: int = 5) -> np.ndarray:
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

    return a0_b + n_bins * (dur_b + n_bins * th_b)


def stratified_kfold_indices(
    p,
    n_folds: int = 5,
    seed: int = 42,
    gamma: float = 1.0,
    n_bins: int = 5,
):
    """
    Возвращает список длины n_folds: каждый элемент — (train_indices, val_indices).
    Внутри каждой страты индексы перемешиваются и равномерно распределяются по фолдам (round-robin).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    rng = np.random.default_rng(seed)
    p = np.asarray(p)
    n = p.shape[0]
    strata = _compute_strata(p, gamma=gamma, n_bins=n_bins)

    val_parts = [[] for _ in range(n_folds)]
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) == 0:
            continue
        perm = rng.permutation(idx)
        for i, j in enumerate(perm):
            val_parts[i % n_folds].append(int(j))

    all_indices = set(range(n))
    folds = []
    for k in range(n_folds):
        val_set = set(val_parts[k])
        val_indices = sorted(val_set)
        train_indices = sorted(all_indices - val_set)
        if not train_indices or not val_indices:
            raise RuntimeError(
                f"Fold {k} has empty train or val (train={len(train_indices)}, val={len(val_indices)}). "
                "Try fewer folds or more samples."
            )
        folds.append((train_indices, val_indices))
    return folds


def fold_sizes(folds):
    return [(len(tr), len(va)) for tr, va in folds]
