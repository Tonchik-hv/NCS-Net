"""Метрики на валидационном фолде (как scripts/evaluate_val_dataset.py)."""
import numpy as np
import torch


def evaluate_val_loader(model, normalizer, val_loader, device):
    model.eval()
    mse_errors = []
    mae_errors = []
    area_errors = []

    with torch.no_grad():
        for p_batch, y_batch, freqs_batch in val_loader:
            p_batch = p_batch.to(device)
            y_batch = y_batch.to(device)
            if freqs_batch.dim() == 1:
                freqs = freqs_batch
            else:
                freqs = freqs_batch[0]

            p_norm = normalizer.norm_params(p_batch)
            y_pred_norm = model(p_norm)
            y_pred_log = normalizer.unnorm_spec(y_pred_norm)

            y_true_np = y_batch.cpu().numpy()
            pred_np = y_pred_log.cpu().numpy()
            if pred_np.ndim == 3:
                pred_np = pred_np.squeeze(1)
            if y_true_np.ndim == 3:
                y_true_np = y_true_np.squeeze(1)

            mse_errors.append(np.mean((y_true_np - pred_np) ** 2, axis=-1))
            mae_errors.append(np.mean(np.abs(y_true_np - pred_np), axis=-1))

            dw = float((freqs[1] - freqs[0]).item())
            recon_spec_log = torch.clamp(y_pred_log, max=3.0)
            if recon_spec_log.dim() == 3:
                recon_spec_log = recon_spec_log.squeeze(1)
            recon_spec_lin = torch.clamp(10 ** recon_spec_log - 1e-9, min=0.0)
            y_lin = torch.clamp(10 ** y_batch.squeeze(1) - 1e-9, min=0.0)

            area_pred = torch.sum(recon_spec_lin, dim=-1) * dw
            area_true = torch.sum(y_lin, dim=-1) * dw
            area_error = torch.abs(area_pred - area_true) / (area_true + 1e-8)
            area_errors.append(area_error.cpu().numpy())

    mse_all = np.concatenate([np.atleast_1d(x) for x in mse_errors])
    mae_all = np.concatenate([np.atleast_1d(x) for x in mae_errors])
    area_all = np.concatenate([np.atleast_1d(x) for x in area_errors])

    return {
        "mse_log_mean": float(np.mean(mse_all)),
        "mse_log_std": float(np.std(mse_all)),
        "mae_log_mean": float(np.mean(mae_all)),
        "mae_log_std": float(np.std(mae_all)),
        "area_rel_error_pct_mean": float(np.mean(area_all) * 100.0),
        "area_rel_error_pct_std": float(np.std(area_all) * 100.0),
        "n_val_samples": int(mse_all.size),
    }
