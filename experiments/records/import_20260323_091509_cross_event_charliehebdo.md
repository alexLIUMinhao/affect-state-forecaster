# Experiment Report: import_20260323_091509_cross_event_charliehebdo

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260323_091509_cross_event_charliehebdo`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260323_091509_cross_event_charliehebdo`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260323_091509_cross_event_charliehebdo
- ratio: unknown
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-23T01:28:01.222173+00:00
- Run root: `runs/20260323_091509_cross_event_charliehebdo`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `affect_state_forecaster`，MAE=0.13929902017116547, RMSE=0.1915578693151474
- 与上一轮相比: 上一轮 `import_20260323_091327_ratio_sweep_ratio_70` 的最优模型是 `thread_transformer_baseline`，本轮最佳 MAE 变化 0.0096。
- `affect_state_forecaster`: status=completed, MAE=0.13929902017116547, RMSE=0.1915578693151474, Pearson=0.2170293096059418, Spearman=0.12298971623903715
- `patchtst_baseline`: status=completed, MAE=0.14247453212738037, RMSE=0.19022923707962036, Pearson=0.23240080330756419, Spearman=0.26586245227734456
- `structure_baseline`: status=completed, MAE=0.15103869140148163, RMSE=0.2162090688943863, Pearson=-0.016577722629256525, Spearman=0.035532725578447125
- `temporal_baseline`: status=completed, MAE=0.16785885393619537, RMSE=0.22112470865249634, Pearson=0.15376721429319554, Spearman=0.11659412946326705
- `text_baseline`: status=completed, MAE=0.20317232608795166, RMSE=0.2551981210708618, Pearson=0.04371378921291583, Spearman=-0.007398220332066588
- `thread_transformer_baseline`: status=completed, MAE=0.14144036173820496, RMSE=0.18748773634433746, Pearson=0.14073667763275885, Spearman=0.16794112067966352

## 图表解读
- Model metrics: `experiments/figures/import_20260323_091509_cross_event_charliehebdo_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `affect_state_forecaster`。
- Event errors: `experiments/figures/import_20260323_091509_cross_event_charliehebdo_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260323_091509_cross_event_charliehebdo_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 affect_state_forecaster，相对文本基线 MAE 改进 0.0639。
- 最佳模型相关系数表现 Pearson=0.2170293096059418, Spearman=0.12298971623903715。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 支持
- Affect-State Forecaster: MAE=0.13929902017116547, RMSE=0.1915578693151474。
- Text baseline: MAE=0.20317232608795166, RMSE=0.2551981210708618。
- Structure baseline: MAE=0.15103869140148163, RMSE=0.2162090688943863。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0353。
- structure_baseline 相对文本基线的 MAE 改进为 0.0521。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### affect_state_forecaster 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### patchtst_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### structure_baseline 的 Pearson/Spearman 接近 0。
- Possible cause: 模型可能只学到了均值附近回归，未捕捉 thread 级差异。
- idea.md conflict: 削弱了“早期观测足以预测未来走势”的论点。
- Recommendation: 需要检查标签噪声、事件切分与模型表达；若多轮如此，再考虑修订研究主线。
### structure_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### temporal_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### text_baseline 的 Pearson/Spearman 接近 0。
- Possible cause: 模型可能只学到了均值附近回归，未捕捉 thread 级差异。
- idea.md conflict: 削弱了“早期观测足以预测未来走势”的论点。
- Recommendation: 需要检查标签噪声、事件切分与模型表达；若多轮如此，再考虑修订研究主线。
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
