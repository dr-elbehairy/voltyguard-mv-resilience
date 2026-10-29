# VoltyGuard — replication package (SASG 2026 full paper)

Code, simulated data generator and results for

> Elbehairy, M., Abu Abuseen, A., Al-Maawi, N., Raouf, A., & Al-Jafin, A. A. (2026). *Resilient Grid Intelligence: A Standards-Guided Graph Neural Network Framework for Incipient Fault Prediction and Cyber-Physical Anomaly Discrimination in Saudi Medium-Voltage Distribution Networks.* Saudi Arabia Smart Grid Conference (SASG 2026), Riyadh, 14–16 December 2026.

## What the package reproduces

| Script | Purpose | Output |
|---|---|---|
| `common.py` | Study design constants, IEEE 33-bus feeder (pandapower `case33bw`) with cable ampacities, IEC 60076-7 thermal model, IEEE C57.104-2019 DGA screening logic, PD / incipient-fault rules (Layer 1), Riyadh summer weather and load models. | — |
| `01_simulate.py` | 24 independent summers (92 days, hourly) with transformer thermal ageing, cable-joint degradation, three kinds of false-data injection and two kinds of sensor malfunction; measured channels with noise; Layer-1 codes and Layer-2 physics residuals computed from measured data only. | `data/real_XX.npz` |
| `02_prepare.py` | Dataset assembly, rolling statistics for the per-asset baseline, normalisation from training realisations, descriptive statistics. | `data/dataset.npz`, `results/results_data.json` |
| `03_models.py --xgb` | Per-asset gradient-boosted trees (XGBoost) with temperature scaling. | `results/preds_xgb.npz` |
| `03_models.py` | VoltyGuard Layer 3 (GRU temporal encoder + 2-layer GraphSAGE-style message passing, two heads) and the two ablations (no graph; no standards/physics features), 3 seeds each. | `results/preds_gnn_*.npz`, `results/results_models.json`, `results/train_log.txt` |
| `04_evaluate.py` | Episode-level recall / precision, false clusters per 100 asset-days, warning lead time, agreement gating, fault-vs-attack discrimination, ECE, SERA action mapping and N-1 transfer feasibility in pandapower. | `results/results_eval.json` |
| `figures.py` | Figures 2–5 (300 dpi). | `figures/` |
| `build_manuscript.py` | Regenerates the manuscript .docx; every number in the paper is read from `results/*.json`. | `paper/VoltyGuard_SASG26_FullPaper.docx` |
| `run_all.py` | Runs everything in order (`--only 04` reruns one step). | — |

`results/` holds the JSON outputs the manuscript was built from, so every number can be traced without rerunning anything.
The simulated arrays (`data/*.npz`, about 0.5 GB) are not committed; `python 01_simulate.py` regenerates them bit-for-bit (seeds 1000–1023) in about 2 minutes on 6 cores.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Python 3.11; torch CPU build is sufficient
python run_all.py                   # ≈ 2 min simulation + ≈ 1 h model training on an Apple M-series laptop
```

Note for macOS: XGBoost and PyTorch are run in separate processes (`--xgb` first) to avoid the duplicate-OpenMP crash.

## Design summary

* Feeder: IEEE 33-bus (Baran & Wu 1989), 12.66 kV, 32 cable sections + 5 ties, one distribution transformer per load bus sized 0.95–1.35 × peak load (0.68–0.84 for the ageing units).
* Climate/load: Riyadh June–August normals (daily mean ≈ 36 °C, 16 K diurnal range, AR(1) day-to-day variation, 2–3 heatwaves of +3–5 K); residential/commercial diurnal shapes; cooling sensitivity +2.5 %/K above 30 °C; backward-forward-sweep power flow every hour.
* Transformer ageing: IEC 60076-7 exponential model (ONAN distribution parameters), relative ageing rate 2^((θ_h − 98)/6); thermal-fault gas generation rising with hot-spot temperature and accelerating with the damage index; C57.104-2019 Tables 1–3 for the Layer-1 status.
* Cable joints: latent damage → Poisson self-clearing faults (location error 20 %), PD level where monitored, contact resistance written into the power-flow model.
* Attacks: provoke (inflated temperature/DGA), stealthy electrical (load and feeding-branch flows scaled consistently with the local voltage drop), mask (15-day replay). Sensor faults: stuck, drift.
* Observability: online DGA on ≈ 50 % of transformers (monthly lab samples elsewhere), PD monitors on ≈ 50 % of joints.
* Evaluation: realisations 0–13 train, 14–17 validation, 18–23 test; alert = calibrated p(failure ≤ 7 d) ≥ τ for 3 consecutive hours (τ from validation F1); lead time = first alert → failure; gating = Layer-1 code ≥ caution within 24 h.

## Licence

Code: MIT (see `LICENSE`). The simulated data carry no third-party rights. pandapower (BSD-3) and its IEEE 33-bus case are used as dependencies.
