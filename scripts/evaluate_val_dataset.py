#!/usr/bin/env python3
"""Оценка прямой генерации на валидационном датасете."""
import os
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from compton_averaging.io import load_for_inference  # noqa: E402


def main():
    device, model, normalizer, val_data = load_for_inference(BASE_DIR)

    # --- ОЦЕНКА НА ВСЕМ ВАЛИДАЦИОННОМ ДАТАСЕТЕ ---
    print("\n--- Оценка прямой генерации на валидационном датасете ---")
    model.eval()
    model.to(device)

    mse_errors = []
    mae_errors = []
    area_errors = []
    inference_times = []

    with torch.no_grad():
        for p_test, y_true, freqs in val_data:
            p_test = p_test.unsqueeze(0).to(device)

            p_norm = normalizer.norm_params(p_test)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            y_pred_norm = model(p_norm)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t_start)

            # Денормализация предсказания (остаемся в логарифмическом масштабе для корректной оценки)
            y_pred_log = normalizer.unnorm_spec(y_pred_norm)

            mse = np.mean((y_true.numpy() - y_pred_log.cpu().numpy())**2)
            mae = np.mean(np.abs(y_true.numpy() - y_pred_log.cpu().numpy()))
            mse_errors.append(mse)
            mae_errors.append(mae)

            dw = float((freqs[1] - freqs[0]).item())
            # Переводим из логарифма
            recon_spec_log = torch.clamp(y_pred_log, max=3.0) # Ограничиваем востановленный логарифмированный спектр, так как инициализированные параметры рандомные, а 10**recon_spec_log может дать взрыв
            recon_spec_lin = torch.clamp(10 ** recon_spec_log - 1e-9, min=0.0)
            y_true_dev = y_true.to(device)
            y_lin = 10 ** y_true_dev - 1e-9

            area_pred = (torch.sum(recon_spec_lin, dim=-1) * dw).cpu().numpy()
            area_true = (torch.sum(y_lin, dim=-1) * dw).cpu().numpy()
            area_error = np.abs(area_pred - area_true) / (area_true + 1e-8)
            area_errors.append(np.mean(area_error))

    print(f"Средняя MSE (log-spectre): {np.mean(mse_errors):.6f}")
    print(f"Средняя MAE (log-spectre): {np.mean(mae_errors):.6f}")
    print(f"Средняя относительная ошибка по площади: {np.mean(area_errors)*100:.2f} %")
    print(f"Общее время инференса (val, {device}): {np.sum(inference_times):.4f} c")
    print(f"Среднее время инференса на 1 сэмпл ({device}): {np.mean(inference_times)*1000:.3f} мс")


if __name__ == "__main__":
    main()
