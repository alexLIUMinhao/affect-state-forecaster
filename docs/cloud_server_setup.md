# Cloud Server Setup

This project can be moved to a Linux GPU server with the repository as the single source of truth.

## Target Layout

```text
/home/alexmhliu/affect-state-forecaster
├── .venv/
├── artifacts/
├── outputs/
├── runs/
└── data/
    ├── raw/
    └── processed/
```

## 1. Clone Repository

## 0. Connect to the Server

Server access currently uses SSH to `root`, then a switch to the project user.

```bash
ssh root@36.138.18.243
su - alexmhliu
cd /home/alexmhliu/affect-state-forecaster
```

For security, plaintext passwords are intentionally not stored in this repository.
Use the currently managed credentials when prompted for `ssh` and `su`.

After switching to `alexmhliu`, all experiment commands below should be run as that user.

```bash
cd /home/alexmhliu
git clone git@github.com:alexLIUMinhao/affect-state-forecaster.git
cd affect-state-forecaster
```

## 2. Create Python Environment

```bash
bash scripts/setup_server_env.sh
source .venv/bin/activate
```

## 3. Check GPU and Data

```bash
python scripts/check_server_env.py
```

This writes `outputs/server_env_report.json` and checks:

- `whoami`
- `uname -a`
- `python3 --version`
- `nvidia-smi`
- `nvcc --version`
- `torch.cuda.is_available()`
- CUDA tensor smoke test
- presence of the `ratio_05` benchmark files

## 4. Place Benchmark Files

For the first-round run, ensure these files exist:

```text
data/processed/pheme_forecast_ratio_05_train.jsonl
data/processed/pheme_forecast_ratio_05_val.jsonl
data/processed/pheme_forecast_ratio_05_test.jsonl
```

If you already have processed files locally, copy them to the same paths on the server before training.

## 5. Run the First-Round 50/50 Experiments

Run all four models:

```bash
bash scripts/run_first_round_experiments.sh
```

Run a subset:

```bash
bash scripts/run_first_round_experiments.sh text_baseline temporal_baseline
```

Useful overrides:

```bash
DEVICE=cuda EPOCHS=10 BATCH_SIZE=32 bash scripts/run_first_round_experiments.sh
```

Outputs are written to:

```text
runs/first_round_ratio_05/<model>/artifacts/
runs/first_round_ratio_05/<model>/eval/
```

Each model directory contains:

- `.pt` weights
- `.json` config
- `_summary.json` training history
- `predictions.jsonl`
- `results_summary.csv`
- `error_analysis_by_event.csv`

## 6. Next Experiments

After the first-round run succeeds, expand to:

- `30/70`, `50/50`, `70/30` observation ratios
- ablations for time, structure, and affect-state
- alternate weak sentiment labelers
- cross-dataset experiments with a new RumourEval pipeline

Recommended next-stage commands on the server:

```bash
source .venv/bin/activate
python scripts/run_ratio_sweep.py --device cuda --epochs 5 --batch_size 16
python scripts/run_affect_ablation_suite.py --device cuda --epochs 5 --batch_size 16
python scripts/run_cross_event_suite.py --device cuda --epochs 5 --batch_size 16
python scripts/run_seed_sweep.py --device cuda --epochs 5 --batch_size 16
python scripts/run_labeler_robustness_suite.py --device cuda --epochs 5 --batch_size 16
```

Tracked experiment runs and generated HTML reports use timestamp-based names by default.

## 7. Experiment Tracking

The server can maintain a unified experiment journal under:

```text
experiments/logs/
experiments/manifests/
experiments/records/
experiments/figures/
```

Import historical results:

```bash
python scripts/sync_experiment_records.py --runs_root runs
```

Run a tracked experiment from scratch:

```bash
python scripts/run_experiment_suite.py \
  --train_path data/processed/pheme_forecast_ratio_05_train.jsonl \
  --val_path data/processed/pheme_forecast_ratio_05_val.jsonl \
  --test_path data/processed/pheme_forecast_ratio_05_test.jsonl \
  --tag ratio05_main
```

Recompute report and figures for an existing manifest:

```bash
python scripts/analyze_experiment_progress.py \
  --run_manifest experiments/manifests/<run_id>.json
```

Primary viewing entry for experiment progress:

```text
experiments/html/index.html
```
