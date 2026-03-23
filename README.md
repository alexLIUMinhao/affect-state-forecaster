# affect-state-forecaster

Research codebase for multimodal event affect forecasting on public-event conversation threads.

## Stage 1 Task

Given the early-stage observations of an event thread, predict the future group sentiment trend.

## Environment

- Python 3.11
- `src/` layout
- `argparse`-based experiment entry points
- Paper V1 scope: semantic + temporal + structural modalities on PHEME
- Server experiment environment: `conda activate asf311`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Data Pipeline

```bash
bash scripts/download_pheme.sh
python scripts/verify_pheme_layout.py
python scripts/prepare_pheme.py
python scripts/label_pheme_sentiment.py
python scripts/build_forecasting_benchmark.py
```

Detailed data instructions are in [docs/data_setup.md](/Users/minhaoliu/Desktop/project/sentimentF/docs/data_setup.md).

Generated processed files:

- `data/processed/pheme_threads.jsonl`
- `data/processed/pheme_threads_labeled.jsonl`
- `data/processed/pheme_forecast_ratio_03.jsonl` and split files
- `data/processed/pheme_forecast_ratio_05.jsonl` and split files
- `data/processed/pheme_forecast_ratio_07.jsonl` and split files

## Training

Train a baseline on the default `50/50` observation split:

```bash
python src/train.py \
  --train_path data/processed/pheme_forecast_ratio_05_train.jsonl \
  --val_path data/processed/pheme_forecast_ratio_05_val.jsonl \
  --model text_baseline
```

Supported models:

- `text_baseline`
- `temporal_baseline`
- `structure_baseline`
- `affect_state_forecaster`

Run the first-round `50/50` experiment suite on a GPU server:

```bash
bash scripts/run_first_round_experiments.sh
```

Run a fully tracked server-side experiment with logs, reports, figures, and `idea.md` alignment analysis:

```bash
python scripts/run_experiment_suite.py \
  --train_path data/processed/pheme_forecast_ratio_05_train.jsonl \
  --val_path data/processed/pheme_forecast_ratio_05_val.jsonl \
  --test_path data/processed/pheme_forecast_ratio_05_test.jsonl \
  --tag ratio05_main
```

Import existing `runs/` results into the experiment journal:

```bash
python scripts/sync_experiment_records.py --runs_root runs
```

Open the HTML progress dashboard after experiments finish:

```bash
open experiments/html/index.html
```

Validate server CUDA and benchmark files before training:

```bash
python scripts/check_server_env.py
```

## Evaluation

```bash
python src/evaluate.py \
  --data_path data/processed/pheme_forecast_ratio_05_test.jsonl \
  --model_path artifacts/text_baseline_pheme_forecast_ratio_05_train.pt \
  --config_path artifacts/text_baseline_pheme_forecast_ratio_05_train.json
```

Evaluation outputs:

- `outputs/predictions.jsonl`
- `outputs/results_summary.csv`
- `outputs/error_analysis_by_event.csv`

For batch runs, per-model outputs are stored under:

- `runs/first_round_ratio_05/<model>/artifacts/`
- `runs/first_round_ratio_05/<model>/eval/`

## Cloud Server Workflow

The deployment target is a Linux GPU server with `/home/alexmhliu/affect-state-forecaster` as the project root.
All server-side experiments should run inside the existing conda environment `asf311`.

Minimal workflow:

```bash
conda activate asf311
cd /home/alexmhliu
git clone git@github.com:alexLIUMinhao/affect-state-forecaster.git
cd affect-state-forecaster
python scripts/check_server_env.py
bash scripts/run_first_round_experiments.sh
```

After each server experiment batch:

```bash
python scripts/sync_experiment_records.py --runs_root runs
python skills/research-progress-html/scripts/generate_progress_html.py \
  --result-source html \
  --output-html experiments/html/$(date +%Y%m%d_%H%M%S)_paper_progress.html
git add experiments/html experiments/figures experiments/manifests experiments/records
git commit -m "Update experiment HTML and records"
git push
```

Detailed server instructions are in [docs/cloud_server_setup.md](/Users/minhaoliu/Desktop/project/sentimentF/docs/cloud_server_setup.md).

## Git Setup

Initialize a local repository after verifying the full pipeline:

```bash
git init -b main
git remote add origin git@github.com:alexLIUMinhao/affect-state-forecaster.git
```

## Notes

- The PHEME benchmark is reconstructed with weak reply-level sentiment labels.
- The primary paper target is thread-level `future_neg_ratio` forecasting.
- Majority-sentiment classification is treated as an auxiliary output for analysis and reporting.

## Model Architecture Diagrams

- [docs/text_baseline_architecture.png](/Users/minhaoliu/Desktop/project/sentimentF/docs/text_baseline_architecture.png)
- [docs/temporal_baseline_architecture.png](/Users/minhaoliu/Desktop/project/sentimentF/docs/temporal_baseline_architecture.png)
- [docs/structure_baseline_architecture.png](/Users/minhaoliu/Desktop/project/sentimentF/docs/structure_baseline_architecture.png)
- [docs/affect_state_forecaster_architecture.png](/Users/minhaoliu/Desktop/project/sentimentF/docs/affect_state_forecaster_architecture.png)
