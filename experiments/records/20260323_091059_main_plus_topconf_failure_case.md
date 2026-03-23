# Failure Case Report: 20260323_091059_main_plus_topconf_failure_case

## 总览
- 分析对象: affect_state_forecaster, patchtst_baseline, thread_transformer_baseline
- 核心结论: affect_state_forecaster 最稳，patchtst_baseline 排序最好，thread_transformer_baseline 有局部交互优势但不够稳。

## 整体指标
- affect_state_forecaster: MAE=0.1052, RMSE=0.1745, Pearson=-0.2058, Spearman=-0.0755
- patchtst_baseline: MAE=0.1055, RMSE=0.1714, Pearson=-0.0270, Spearman=0.3499
- thread_transformer_baseline: MAE=0.1184, RMSE=0.1773, Pearson=-0.1120, Spearman=0.3212

## 典型案例数量
- shared_failures: 10
- affect_vs_patch_conflicts: 10
- thread_transformer_unique_wins: 0
- flip_up_cases: 10
