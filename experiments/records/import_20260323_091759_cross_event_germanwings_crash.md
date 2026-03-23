# Experiment Report: import_20260323_091759_cross_event_germanwings_crash

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260323_091759_cross_event_germanwings_crash`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260323_091759_cross_event_germanwings_crash`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260323_091759_cross_event_germanwings_crash
- ratio: unknown
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-23T01:28:03.070587+00:00
- Run root: `runs/20260323_091759_cross_event_germanwings_crash`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `structure_baseline`，MAE=0.11534103751182556, RMSE=0.225380077958107
- 与上一轮相比: 上一轮 `import_20260323_091703_cross_event_ferguson` 的最优模型是 `thread_transformer_baseline`，本轮最佳 MAE 变化 0.0082。
- `affect_state_forecaster`: status=completed, MAE=0.11933886259794235, RMSE=0.21641682088375092, Pearson=0.18204722929834202, Spearman=0.5712503945452716
- `patchtst_baseline`: status=completed, MAE=0.124762624502182, RMSE=0.21938441693782806, Pearson=0.04135399575027687, Spearman=0.3435150231229094
- `structure_baseline`: status=completed, MAE=0.11534103751182556, RMSE=0.225380077958107, Pearson=-7.837074425378712e-05, Spearman=0.37337212308053014
- `temporal_baseline`: status=completed, MAE=0.17835037410259247, RMSE=0.25406599044799805, Pearson=-0.02809648398223509, Spearman=-0.07252552031060658
- `text_baseline`: status=completed, MAE=0.16604608297348022, RMSE=0.2797043025493622, Pearson=-0.4619664704101862, Spearman=-0.21175661177109203
- `thread_transformer_baseline`: status=completed, MAE=0.1178584173321724, RMSE=0.2183990776538849, Pearson=0.07865833701176773, Spearman=0.473206635606859

## 图表解读
- Model metrics: `experiments/figures/import_20260323_091759_cross_event_germanwings_crash_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `structure_baseline`。
- Event errors: `experiments/figures/import_20260323_091759_cross_event_germanwings_crash_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260323_091759_cross_event_germanwings_crash_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 structure_baseline，相对文本基线 MAE 改进 0.0507。
- 最佳模型相关系数表现 Pearson=-7.837074425378712e-05, Spearman=0.37337212308053014。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 部分支持
- Affect-State Forecaster: MAE=0.11933886259794235, RMSE=0.21641682088375092。
- Text baseline: MAE=0.16604608297348022, RMSE=0.2797043025493622。
- Structure baseline: MAE=0.11534103751182556, RMSE=0.225380077958107。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 -0.0123。
- structure_baseline 相对文本基线的 MAE 改进为 0.0507。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### temporal_baseline 未优于 text_baseline。
- Possible cause: 当前时序编码没有有效利用 reply 顺序，或 observation window 信息密度不足。
- idea.md conflict: 削弱了 `idea.md` 中时间动态应带来增益的预期。
- Recommendation: 优先检查时间编码与序列长度处理；单次出现不建议新建 `idea.md`。
### affect_state_forecaster 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### patchtst_baseline 的 Accuracy 远高于 Macro-F1。
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
### thread_transformer_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
