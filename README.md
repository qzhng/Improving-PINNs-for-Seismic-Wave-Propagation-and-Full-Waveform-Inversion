# Improving PINNs for Seismic Wave Propagation and Full Waveform Inversion


This repository reproduces and extends the PINN-based seismic inversion framework from:

Rasht-Behesht, M., Huber, C., Shukla, K., & Karniadakis, G. E. (2022). *Physics-Informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions.* Journal of Geophysical Research: Solid Earth, 127.

The final written report is in `tex/report-final.pdf` (source: `tex/report-final.tex`).

## What this project does

We solve an acoustic-wave inverse problem in 2D:

- Governing PDE: $\alpha^2(x,z)\nabla^2\phi = \partial_t^2\phi$
- Unknown to recover: wavespeed field $\alpha(x,z)$
- Data used for inversion:
  - two early-time wavefield snapshots
  - sparse surface seismograms from 20 receivers

Compared to the reference paper, this repo emphasizes reproducibility and controlled ablation:

- Replaces SpecFem2D ground truth generation with a self-contained finite-difference solver.
- Reproduces a Case-3-like ellipsoidal anomaly inversion setup.
- Tests four targeted PINN improvements one-at-a-time against a shared baseline.

Project goal:

- Build an end-to-end, reproducible PINN inversion pipeline where data generation, training, and analysis are all local to this repository.
- Identify which PINN improvements actually help on this seismic FWI task under controlled conditions.
- Separate "theory says this should help" from "it improves this specific coupled inverse optimization in practice."

## Core methodology (final report)

### Ground truth data generation

`code/fd_solver.py` generates synthetic training/validation data with:

- 2nd-order central finite differences in space
- explicit leapfrog time stepping
- CFL-stable time step
- sponge absorbing layers (left/right/bottom)
- stress-free top boundary via image method
- 20 Hz Ricker source

Data products used by training:

- two early-time snapshots at $t_1=0.100$ s and $t_2=0.115$ s (used to encode source/initial wavefield information)
- 20 surface receiver traces over 0.4 s
- PDE collocation point pools sampled in space-time

Why a finite-difference generator is sufficient here:

- The target geometries are synthetic and relatively simple (regular domain, smooth/piecewise-smooth velocity field).
- Accuracy can be controlled by grid and time-step selection under the CFL condition.
- The project focus is PINN training behavior, not high-order production wave simulation.

### PINN formulation

Two networks are trained jointly:

- WavefieldNet: $(x,z,t) \rightarrow \phi$
- WavespeedNet: $(x,z) \rightarrow \alpha$

Loss terms:

- PDE residual loss
- snapshot fitting loss
- surface observation (seismogram) loss
- free-surface boundary loss

Important implementation detail from the report:

- Inputs are normalized to [-1, 1], and the PDE residual includes chain-rule scaling factors for $x,z,t$. Omitting these factors causes incorrect convergence.

Wavespeed parameterization details:

- The inverse model predicts a bounded perturbation around a background velocity.
- A smooth spatial mask restricts updates mostly to the interior, reducing boundary artifacts.

Training objective:

- Both networks are optimized jointly with Adam.
- The objective balances physics consistency (PDE + free-surface terms) and data consistency (snapshots + seismograms).

## Improvements evaluated

All experiments use the same inverse setup and 30,000 epochs (single RTX 4070 in report runs), modifying only one component at a time.

1. Adaptive loss weighting (Wang et al., 2021)
- Dynamically updates loss weights based on gradient statistics.
- Motivation: fixed manual weights can cause one loss to dominate the gradient flow.
- In-code behavior: weights are updated periodically using gradient-magnitude ratios with smoothing.

2. Activation study
- tanh baseline
- sin (SIREN-style)
- adaptive tanh (trainable slope)
- Motivation: PINNs are frequency-sensitive; activation choice strongly affects high-frequency recovery and stability.

3. Random Fourier features
- Input encoding to reduce spectral bias.
- Motivation: enrich coordinate representation with tunable frequency bandwidth.
- Caveat seen in this project: performance is sensitive to bandwidth mismatch.

4. Residual-based adaptive collocation
- Periodically resamples PDE points, prioritizing high-residual regions.
- Motivation: spend PDE supervision budget where the current model violates physics the most.

Controlled experiment design:

- Same dataset, random seeds, and optimization schedule for all runs.
- One component changed at a time relative to baseline.
- 30,000 epochs per run in the final report.

## Final quantitative results (from report)

Case: ellipsoidal low-velocity anomaly in a 3.0 km/s background.

| Experiment | Wavespeed RMSE (km/s) | Wavefield relative error | Time (s) |
|---|---:|---:|---:|
| Baseline (tanh, fixed weights) | 0.190 | 0.875 | 1212 |
| Adaptive weights | 0.187 | 0.898 | 1224 |
| Sin activation (SIREN) | 0.126 | 0.909 | 1209 |
| Adaptive tanh activation | 1.119 | 1.064 | 1298 |
| Fourier features | 0.229 | 0.990 | 1340 |
| Adaptive collocation | 0.125 | 1.026 | 1162 |

Key findings:

- Best wavespeed recovery: sin activation and adaptive collocation (both about 34% RMSE reduction vs baseline).
- Adaptive weighting: only marginal gain in this setup.
- Adaptive tanh: catastrophic failure mode in the coupled inverse setting.
- Fourier features: degraded performance for the chosen bandwidth.
- Important nuance: better wavespeed RMSE did not imply better wavefield error.

Interpretation from the report discussion:

- Sin activations and adaptive collocation improved velocity recovery through different mechanisms.
- Adaptive tanh failed due to saturation/vanishing-gradient dynamics in the coupled inverse setting.
- Fourier features likely underperformed because the selected feature bandwidth did not align with dominant wavefield/anomaly frequencies.

Main scientific takeaway:

- Better inversion quality (velocity RMSE) does not automatically translate to better reconstructed wavefield amplitude/phase.
- The coupled optimization has a moving-target effect: as velocity changes, the wavefield network must continuously re-adapt.

## Repository structure

- `code/fd_solver.py`: finite-difference wave solver used for synthetic data
- `code/data_loader.py`: loads snapshots, seismograms, collocation pools
- `code/pinn_core.py`: MLPs, activations, Fourier encoding
- `code/train_pinn.py`: PINN model, losses, trainer
- `code/run_experiment.py`: runs baseline + improvement experiments, saves figures/summary
- `code/analyze_results.py`: post-processing and publication-style plots/tables
- `code/quick_test.py`: smoke tests for imports, model/loss pipeline, short training checks
- `results/`: generated outputs
- `tex/report-final.tex` and `tex/report-final.pdf`: final write-up

Typical outputs after a full run:

- `results/<experiment>/history.npy`: per-epoch metrics
- `results/<experiment>/` plots/checkpoints for each experiment
- `results/summary/summary.csv`: aggregate table used in analysis
- `results/summary/` comparison plots (RMSE bars, loss curves, model comparisons)
- `results/analysis/` publication-style figures and LaTeX table fragments

Experiment names used in code:

- baseline
- adaptive_weights
- activation_sin
- activation_adaptive
- fourier_features
- adaptive_colloc
- best_combined (extra exploratory run in script)

Note: the final report's main comparison focuses on the first six experiments.

## Experimental setup used in the report

- Domain: 1.4 km x 0.5 km
- Background velocity: 3.0 km/s
- Target anomaly: ellipsoidal low-velocity inclusion (2.5 km/s)
- Source: 20 Hz Ricker wavelet
- Receiver geometry: 20 uniformly spaced surface receivers
- Epochs: 30,000
- Optimizer: Adam with learning-rate decay
- Hardware in report: single NVIDIA RTX 4070

Shared baseline loss weights:

- $\lambda_{pde}=0.1$
- $\lambda_{snap}=1.0$
- $\lambda_{obs}=1.0$
- $\lambda_{fs}=0.1$

Why this setup matters:

- It provides a fixed testbed so performance differences can be attributed to method changes rather than confounders.

## How to run

From the `code/` directory:

Quick smoke test:

```bash
python quick_test.py --device cuda
```

Quick experiment run (500 epochs):

```bash
python run_experiment.py --device cuda --quick --out_dir ../results_quick
```

Full experiment run:

```bash
python run_experiment.py --device cuda --epochs 30000 --out_dir ../results
```

Post-run analysis:

```bash
python analyze_results.py --out_dir ../results
```

If CUDA is unavailable, switch `--device cpu`.

Recommended workflow:

1. Run `quick_test.py` first to verify imports, autograd paths, and a short trainer loop.
2. Run a quick 500-epoch experiment to check end-to-end plumbing.
3. Launch the full 30k run once the quick run looks healthy.
4. Generate analysis figures/tables.

Expected runtime:

- Quick mode: minutes.
- Full mode: roughly 1-3 hours on a modern single GPU (depends on hardware and CUDA setup).

## Notes on reproducibility and interpretation

- Absolute numbers should not be compared directly to the reference paper because solver/training setups differ (SpecFem2D+PML vs FD+sponge, architecture/training schedule differences).
- The report focuses on controlled relative comparisons under a fixed local setup.
- A natural next step (as noted in the report) is combining the two strongest ideas (sin activations + adaptive collocation) and retuning Fourier bandwidth around a physics-informed estimate.

Additional limitations recorded in the report:

- Finite-difference data may include numerical dispersion characteristics different from a spectral-element solver.
- Network width/depth and training horizon differ from the original paper.
- Results are based on one synthetic configuration, so generalization claims should be cautious.

Suggested next experiments:

1. Combine sin activation + adaptive collocation as the primary hybrid baseline.
2. Retune Fourier feature bandwidth using a physics-informed estimate (rather than a generic default).
3. Try staged training (forward fit first, then inverse coupling) to reduce the velocity-wavefield error gap.
4. Scale network size and compare whether ranking of improvements remains stable.

## Citation references used in the final report

- Raissi et al. (2019) PINNs
- Rasht-Behesht et al. (2022) seismic PINNs/FWI
- Wang et al. (2021) adaptive loss balancing
- Sitzmann et al. (2020) SIREN
- Jagtap et al. (2020) adaptive activations
- Tancik et al. (2020) Fourier features
