#!/usr/bin/env python3
"""Замер времени инференса модели на CPU."""
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

    # --- ЗАМЕР ВРЕМЕНИ ИНФЕРЕНСА НА CPU ---
    print("\n--- Замер времени инференса модели на CPU ---")
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    normalizer.cpu()
    model_cpu.eval()
    cpu_inference_times = []
    with torch.no_grad():
        for p_test, _, _ in val_data:
            p_norm = normalizer.norm_params(p_test.unsqueeze(0).to(cpu_device))
            t_start = time.perf_counter()
            _ = model_cpu(p_norm)
            cpu_inference_times.append(time.perf_counter() - t_start)
    print(f"Среднее время инференса на CPU (1 сэмпл): {np.mean(cpu_inference_times)*1000:.3f} мс")
    model.to(device)
    normalizer.to(device)


if __name__ == "__main__":
    main()
