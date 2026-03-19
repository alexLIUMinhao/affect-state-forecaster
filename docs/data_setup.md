# PHEME Data Setup

This project expects the raw PHEME dataset under:

```text
data/raw/pheme/
```

Processed outputs:

```text
data/processed/pheme_threads.jsonl
data/processed/pheme_threads_labeled.jsonl
data/processed/pheme_forecast_ratio_03.jsonl
data/processed/pheme_forecast_ratio_05.jsonl
data/processed/pheme_forecast_ratio_07.jsonl
```

## Expected Workflow

```bash
bash scripts/download_pheme.sh
python scripts/verify_pheme_layout.py
python scripts/prepare_pheme.py
python scripts/label_pheme_sentiment.py
python scripts/build_forecasting_benchmark.py
```

## Raw Layout

The verifier and preprocessor expect:

```text
data/raw/pheme/
└── <event_name>/
    └── <thread_id>/
        ├── source-tweets/
        │   └── *.json
        ├── reactions/
        │   └── *.json
        └── structure.json
```

## Download Options

Automatic download:

```bash
PHEME_URL="https://example.com/pheme.zip" bash scripts/download_pheme.sh
```

Manual archive placement:

```bash
ARCHIVE_PATH="/absolute/path/to/pheme.zip" bash scripts/download_pheme.sh
```

Manual extracted placement:

- Copy extracted event folders directly into `data/raw/pheme/`
- Then run the workflow shown above

## Verification

```bash
python scripts/verify_pheme_layout.py
```

The verifier checks:

- event and thread directories are discoverable
- `source-tweets/` exists and contains at least one JSON file
- `reactions/` exists
- `structure.json` exists
- sampled JSON files are parseable

Use strict mode to fail on warnings:

```bash
python scripts/verify_pheme_layout.py --strict
```

## Preprocessing

```bash
python scripts/prepare_pheme.py
```

This writes one normalized thread record per line to `data/processed/pheme_threads.jsonl` with:

- `thread_id`
- `event_name`
- `source_text`
- `source_created_at`
- `conversation_tree`
- `replies`

Replies are kept in chronological order and preserve `parent_id`.

## Weak Sentiment Labeling

```bash
python scripts/label_pheme_sentiment.py
```

This writes `data/processed/pheme_threads_labeled.jsonl` and assigns each reply a weak 3-way sentiment label:

- `negative`
- `neutral`
- `positive`

## Forecast Benchmark Construction

```bash
python scripts/build_forecasting_benchmark.py
```

This materializes three observation settings:

- `30/70`
- `50/50`
- `70/30`

For each setting it writes:

- one full JSONL
- one `_train.jsonl`
- one `_val.jsonl`
- one `_test.jsonl`

Each forecast sample includes:

- `thread_id`
- `event_name`
- `split`
- `observation_ratio`
- `source_text`
- `conversation_tree`
- `observed_replies`
- `forecast_replies`
- observed sentiment summary
- future sentiment summary

## Failure Handling

- If no raw threads are found, preprocessing writes an empty JSONL file and logs guidance.
- If a thread is malformed, preprocessing logs a warning and skips that thread.
- If a JSON file is malformed, the verifier reports it explicitly so the raw dataset can be fixed or replaced.
