# Writing Rules

## Introduction

The introduction has two paragraphs.

Paragraph 1:
- State the research goal.
- State the forecasting task.
- Mention the current benchmark context, usually PHEME.

Paragraph 2:
- Always cite the current best model.
- Always mention MAE gain versus the text baseline when available.
- If `affect_state_forecaster` is best, emphasize preliminary support for the affect-state hypothesis.
- If `structure_baseline` or `temporal_baseline` is best, emphasize that the task is valid and multimodal cues help, but the current method still needs diagnosis.
- If evidence is incomplete, use conservative wording and avoid superiority claims.

## Main experiment progress

Use a `Table 1` layout with complete structure even when values are missing.

Required columns:
- `Dataset`
- `Compared Baselines`
- `Ours`
- `Metrics`

Metric string order:
- `MAE`
- `RMSE`
- `Pearson`
- `Spearman`

## Next-step planning

- If `affect_state_forecaster` is best: prioritize multi-ratio, cross-event, and robustness experiments.
- If the task is supported but `affect_state_forecaster` is not best: prioritize diagnosis, ablation, regularization, and fusion refinement.
- If evidence is insufficient: prioritize result collection and structured result files before model expansion.
