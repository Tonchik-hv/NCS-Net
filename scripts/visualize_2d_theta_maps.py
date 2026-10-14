#!/usr/bin/env python3
"""2D визуализация предсказаний I(theta, omega) и связанные метрики."""
import os
import sys

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from compton_averaging.io import load_for_inference  # noqa: E402
from compton_averaging.viz_2d_theta_maps import run_2d_theta_maps  # noqa: E402

plt.style.use('physrev.mplstyle')
plt.rcParams['figure.dpi'] = "300"


def main():
    device, model, normalizer, val_data = load_for_inference(BASE_DIR)
    run_2d_theta_maps(device, model, normalizer, val_data, base_dir=BASE_DIR)


if __name__ == "__main__":
    main()
