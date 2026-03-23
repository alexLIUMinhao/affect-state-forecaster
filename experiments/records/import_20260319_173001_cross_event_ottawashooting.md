# Experiment Report: import_20260319_173001_cross_event_ottawashooting

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260319_173001_cross_event_ottawashooting`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260319_173001_cross_event_ottawashooting`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260319_173001_cross_event_ottawashooting
- ratio: unknown
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T09:44:41.494206+00:00
- Run root: `runs/20260319_173001_cross_event_ottawashooting`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `temporal_baseline`，MAE=0.12836137413978577, RMSE=0.17434574663639069
- 与上一轮相比: 上一轮 `import_20260319_172927_cross_event_germanwings_crash` 的最优模型是 `structure_baseline`，本轮最佳 MAE 变化 0.013。
- `affect_state_forecaster`: status=completed, MAE=0.13433682918548584, RMSE=0.1693952977657318, Pearson=0.1252576525749988, Spearman=0.13100225584311334
- `structure_baseline`: status=completed, MAE=0.13225902616977692, RMSE=0.17402568459510803, Pearson=-0.012561287181751295, Spearman=0.22853886990971087
- `temporal_baseline`: status=completed, MAE=0.12836137413978577, RMSE=0.17434574663639069, Pearson=0.20398168752550228, Spearman=0.07694237297335697
- `text_baseline`: status=completed, MAE=0.1787051558494568, RMSE=0.22408145666122437, Pearson=0.0402426033273651, Spearman=0.13801001843734104

## 图表解读
- Model metrics: `experiments/figures/import_20260319_173001_cross_event_ottawashooting_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `temporal_baseline`。
- Event errors: `experiments/figures/import_20260319_173001_cross_event_ottawashooting_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260319_173001_cross_event_ottawashooting_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 temporal_baseline，相对文本基线 MAE 改进 0.0503。
- 最佳模型相关系数表现 Pearson=0.20398168752550228, Spearman=0.07694237297335697。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 部分支持
- Affect-State Forecaster: MAE=0.13433682918548584, RMSE=0.1693952977657318。
- Text baseline: MAE=0.1787051558494568, RMSE=0.22408145666122437。
- Structure baseline: MAE=0.13225902616977692, RMSE=0.17402568459510803。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0503。
- structure_baseline 相对文本基线的 MAE 改进为 0.0464。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### affect_state_forecaster 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### structure_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### temporal_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### text_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
