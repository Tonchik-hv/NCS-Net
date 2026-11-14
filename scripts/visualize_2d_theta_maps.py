#!/usr/bin/env python3
"""2D визуализация предсказаний I(theta, omega) и связанные метрики."""
import argparse
import os
import sys

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from compton_averaging.io import load_for_inference  # noqa: E402
from compton_averaging.viz_2d_theta_maps import run_2d_theta_maps  # noqa: E402

_physrev_style = os.path.join(BASE_DIR, "physrev.mplstyle")
if not os.path.isfile(_physrev_style):
    raise FileNotFoundError(
        f"Не найден файл стиля: {_physrev_style}\n"
        "Он должен лежать в корне репозитория (см. README)."
    )
plt.style.use(_physrev_style)
plt.rcParams["figure.dpi"] = "300"


def main():
    parser = argparse.ArgumentParser(
        description="2D визуализация предсказаний I(theta, omega).",
    )
    parser.add_argument(
        "--zoom-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inset на predicted/analytical lab-картах: линейный график спектра, интегрированного по углу "
            "в полосе y∈[0, γ·θ_c]; по x — [0.2, 1.8] (ось y_lab как на карте)"
        ),
    )
    args = parser.parse_args()

    device, model, normalizer, val_data = load_for_inference(BASE_DIR)
    run_2d_theta_maps(
        device,
        model,
        normalizer,
        val_data,
        base_dir=BASE_DIR,
        zoom_box=args.zoom_box,
    )


if __name__ == "__main__":
    main()
