# multimodal-event-affect-forecasting

Research codebase for multimodal event affect forecasting on public-event conversation threads.

## Stage 1 Task

Given the early-stage observations of an event thread, predict the future group sentiment trend.

## Environment

- Python 3.11
- `src/` layout
- `argparse`-based experiment entry points
- Paper V1 scope: semantic + temporal + structural modalities on PHEME

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
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

## Git Setup

Initialize a local repository after verifying the full pipeline:

```bash
git init -b main
git remote add origin https://github.com/<your-username>/multimodal-event-affect-forecasting.git
```

## Notes

- The PHEME benchmark is reconstructed with weak reply-level sentiment labels.
- The primary paper target is thread-level `future_neg_ratio` forecasting.
- Majority-sentiment classification is treated as an auxiliary output for analysis and reporting.
