# Experiment Report: import_20260319_172820_cross_event_ebola_essien

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260319_172820_cross_event_ebola_essien`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260319_172820_cross_event_ebola_essien`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260319_172820_cross_event_ebola_essien
- ratio: unknown
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T09:44:39.857495+00:00
- Run root: `runs/20260319_172820_cross_event_ebola_essien`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `temporal_baseline`，MAE=0.05972917005419731, RMSE=0.0698486864566803
- 与上一轮相比: 上一轮 `import_20260319_172746_cross_event_charliehebdo` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 -0.0834。
- `affect_state_forecaster`: status=completed, MAE=0.09168344736099243, RMSE=0.09815613925457001, Pearson=-0.9999999999999999, Spearman=-0.9999999999999999
- `structure_baseline`: status=completed, MAE=0.09576581418514252, RMSE=0.11622308939695358, Pearson=-1.0, Spearman=-0.9999999999999999
- `temporal_baseline`: status=completed, MAE=0.05972917005419731, RMSE=0.0698486864566803, Pearson=1.0, Spearman=0.9999999999999999
- `text_baseline`: status=completed, MAE=0.31166326999664307, RMSE=0.3416330814361572, Pearson=-1.0, Spearman=-0.9999999999999999

## 图表解读
- Model metrics: `experiments/figures/import_20260319_172820_cross_event_ebola_essien_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `temporal_baseline`。
- Event errors: `experiments/figures/import_20260319_172820_cross_event_ebola_essien_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260319_172820_cross_event_ebola_essien_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 temporal_baseline，相对文本基线 MAE 改进 0.2519。
- 最佳模型相关系数表现 Pearson=1.0, Spearman=0.9999999999999999。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 部分支持
- Affect-State Forecaster: MAE=0.09168344736099243, RMSE=0.09815613925457001。
- Text baseline: MAE=0.31166326999664307, RMSE=0.3416330814361572。
- Structure baseline: MAE=0.09576581418514252, RMSE=0.11622308939695358。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.2519。
- structure_baseline 相对文本基线的 MAE 改进为 0.2159。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
- No material mismatch signals detected in this run.

## 下一步决策
- 决策动作: 继续主线
- 原因: 当前结果减少了不确定性，且没有明显削弱核心研究叙事。
- 因此下一步是 进入下一个主实验包，例如多 ratio 或泛化实验。，而不是直接扩大实验范围。
