# NCS-Net

**NCS-Net** is a neural network for direct generation of averaged nonlinear Compton scattering spectra from parameters `(a₀, τ, θ)`. This repository contains training, evaluation, visualization, and inference code for arbitrary parameters and for an electron beam.

The original monolithic script `NN_for_averaging_full_param_angles_pi_2_inference_time.py` is kept unchanged; for day-to-day use from this repository, prefer the modular layout below.

## Repository layout

```
.
├── train.py                          # NCS-Net training (main entry point)
├── physrev.mplstyle                  # matplotlib plot style
│
├── compton_averaging/                # Python package
│   ├── config.py                     # hyperparameters and artifact paths
│   ├── dataset.py                    # dataset generation / loading
│   ├── model.py                      # NCS-Net architecture (DirectComptonModel)
│   ├── normalizer.py                 # parameter and spectrum normalization
│   ├── loss.py                       # physics-informed loss
│   ├── training.py                   # training and validation loop
│   ├── physics.py                    # snlcs calculator initialization
│   ├── plotting.py                   # plotting utilities
│   ├── io.py                         # load NCS-Net, normalizer, val split
│   └── viz_2d_theta_maps.py          # 2D maps I(θ, ω) and metrics
│
├── scripts/                          # standalone scripts
│   ├── evaluate_val_dataset.py       # metrics on the validation set
│   ├── benchmark_cpu_inference.py    # 1D inference timing on CPU
│   ├── visualize_2d_theta_maps.py    # 2D maps and 2D metrics
│   ├── visualize_random_predictions.py  # N random 1D plots
│   ├── predict_spectrum_from_params.py  # inference for one (a₀, τ, θ) triple
│   ├── predict_beam_spectrum_corrected.py  # 2D beam spectrum (NCS-Net / analytic / quad)
│   └── benchmark_beam_spectrum_cpu.py  # 2D beam spectrum CPU benchmark
│
├── src/
│   └── calc_spectrum.py              # analytic and numerical 1D spectra
│
└── config/
    ├── ExtFieldsParams.yaml      # laser field parameters
    ├── eBeamParams.yaml          # electron beam parameters
    └── OtherParams.yaml          # other snlcs parameters
```

Root-level `predict_spectrum_from_params.py`, `predict_beam_spectrum_corrected.py`, and `benchmark_beam_spectrum_cpu.py` are **legacy sources** kept as-is. For publication and runs from this repo, use the copies under `scripts/`.

## Installation

```bash
# from the repository root
pip install torch numpy scipy matplotlib pyyaml

# snlcs library (required for quad_interp and some visualizations)
pip install -e snlcs/
```

All commands below are run **from the repository root**.

## Preparation: training NCS-Net

Before evaluation and inference you need a trained NCS-Net and a normalizer. On the first run, the dataset is also generated (or loaded from cache).

```bash
python train.py
```

The script:
- creates/loads the dataset cache in `compton_dataset_cache_angles_pi_2_ns800000_ts2500_seed42/`;
- builds a stratified train/val split and saves indices to `val_indices_angles_pi_2_ns800000_ts2500.pkl`;
- trains the normalizer → `normalizer_angles_pi_2_ns800000_ts2500.pkl`;
- trains NCS-Net (if the checkpoint does not exist yet) → `model_with_averaging_angles_pi_2_ns800000_ts2500.pth`.

If NCS-Net is already trained, `train.py` only loads the weights and does not retrain.

---

## Scripts: purpose and usage

### Training

| File | Purpose | Run |
|------|---------|-----|
| `train.py` | Dataset generation, normalizer and NCS-Net training, saving weights | `python train.py` |

### Evaluating the trained NCS-Net (validation set)

Requires: `model_with_averaging_angles_pi_2_ns800000_ts2500.pth`, `normalizer_angles_pi_2_ns800000_ts2500.pkl`, and the dataset cache.

| File | Purpose | Run |
|------|---------|-----|
| `scripts/evaluate_val_dataset.py` | MSE/MAE in log scale, area error, GPU/CPU inference time | `python scripts/evaluate_val_dataset.py` |
| `scripts/benchmark_cpu_inference.py` | Average time of one forward pass on CPU | `python scripts/benchmark_cpu_inference.py` |
| `scripts/visualize_2d_theta_maps.py` | 2D maps I(θ, ω): prediction, analytic, Lab frame; 2D val metrics | `python scripts/visualize_2d_theta_maps.py` |
| `scripts/visualize_random_predictions.py` | N random 1D plots: True vs Predicted vs snlcs | `python scripts/visualize_random_predictions.py` |

### Inference for arbitrary parameters

| File | Purpose | Run |
|------|---------|-----|
| `scripts/predict_spectrum_from_params.py` | Predict a 1D spectrum for `(a₀, τ, θ)` with True vs Predicted plot | `python scripts/predict_spectrum_from_params.py --a0 0.8 --tau 50 --theta 2.5` |

Useful options:

```bash
python scripts/predict_spectrum_from_params.py --a0 1.0 --tau 40 --theta 3.0 -o output.png
python scripts/predict_spectrum_from_params.py --a0 0.5 --tau 30 --theta 2.8 --skip-snlcs
python scripts/predict_spectrum_from_params.py --help
```

Parameters:
- `--a0` — field amplitude;
- `--tau` — pulse duration in units of π (`dur = tau * π`);
- `--theta` — angle in radians (training range: π/2 … π);
- `--model`, `--normalizer` — paths to weights (defaults from `compton_averaging/config.py`);
- `--output` / `-o` — path to PNG;
- `--skip-snlcs` — do not call snlcs (faster if the package is unavailable).

### Inference for an electron beam

Uses `config/ExtFieldsParams.yaml` and `config/eBeamParams.yaml`.

| File | Purpose | Run |
|------|---------|-----|
| `scripts/predict_beam_spectrum_corrected.py` | Three 2D beam spectrum maps: NCS-Net (with Lorentz transform), D2ECalculator_angles analytic, snlcs quad_interp; integral metrics | `python scripts/predict_beam_spectrum_corrected.py` |
| `scripts/benchmark_beam_spectrum_cpu.py` | CPU benchmark: NCS-Net vs D2ECalculator_angles for 2D beam spectrum (no plots) | `python scripts/benchmark_beam_spectrum_cpu.py` |

Output PNGs (`predict_beam_spectrum_corrected.py`):
- `predicted_beam_spectrum_map.png` — NCS-Net + Lorentz transform
- `analytical_beam_spectrum_map.png` — D2ECalculator_angles analytic
- `analytical_beam_spectrum_theta_frequency_map_lab.png` — snlcs quad_interp (third map)

---

## Module dependencies

```
src/calc_spectrum.py
  └── used in: dataset, viz_2d, predict_spectrum_from_params, predict_beam_spectrum_corrected

config/*.yaml
  └── used in: physics.py, predict_beam_spectrum_corrected, benchmark_beam_spectrum_cpu, visualize_random_predictions

snlcs (pip install -e snlcs/)
  └── required for: quad_interp spectra, init_spectrum_sqinrp, predict_beam_spectrum_corrected
```

## snlcs configuration files

| File | Contents |
|------|----------|
| `config/ExtFieldsParams.yaml` | Field type, `a₀`, wavelength, pulse duration `T` |
| `config/eBeamParams.yaml` | Gaussian beam: energy, sizes, number of particles |
| `config/OtherParams.yaml` | Integration method, frequency/angle grid, `random_seed` |

Before running beam scripts, edit `a₀`, `T`, and beam parameters in these YAML files if needed.

## Artifacts (created at runtime)

| File / directory | Description |
|------------------|-------------|
| `model_with_averaging_angles_pi_2_ns800000_ts2500.pth` | Trained NCS-Net weights |
| `normalizer_angles_pi_2_ns800000_ts2500.pkl` | Normalization parameters |
| `val_indices_angles_pi_2_ns800000_ts2500.pkl` | Validation split indices |
| `compton_dataset_cache_angles_pi_2_ns.../` | Generated dataset cache |
| `training_history_with_averaging.png` | Train/val loss curves |
| `val_2d_metrics_maps/` | Random 2D maps from validation metrics |
| `averaged_prediction_result_angles_pi_2_*.png` | Random 1D plots from visualization |
| `predicted_spectrum_theta_frequency_map*.png` | 2D maps from `visualize_2d_theta_maps.py` |

## Typical workflow

```bash
# 1. Install dependencies
pip install torch numpy scipy matplotlib pyyaml
pip install -e snlcs/

# 2. Train (or load existing weights)
python train.py

# 3. Validation evaluation
python scripts/evaluate_val_dataset.py
python scripts/visualize_random_predictions.py

# 4. Inference for specific parameters
python scripts/predict_spectrum_from_params.py --a0 0.8 --tau 50 --theta 2.8

# 5. Beam spectrum (after tuning config/*.yaml)
python scripts/predict_beam_spectrum_corrected.py
python scripts/benchmark_beam_spectrum_cpu.py
```

## `compton_averaging` package

Common imports for your own scripts (`DirectComptonModel` is the NCS-Net implementation):

```python
from compton_averaging import DirectComptonModel, ComptonDataset, train_model  # DirectComptonModel = NCS-Net
from compton_averaging.io import load_for_inference
from compton_averaging.config import model_file, normalizer_file, DATASET_NUM_SAMPLES
```
