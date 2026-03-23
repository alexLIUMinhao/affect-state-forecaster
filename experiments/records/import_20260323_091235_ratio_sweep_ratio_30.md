# Experiment Report: import_20260323_091235_ratio_sweep_ratio_30

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260323_091235_ratio_sweep_ratio_30`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260323_091235_ratio_sweep_ratio_30`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260323_091235_ratio_sweep_ratio_30
- ratio: unknown
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-23T01:27:59.926827+00:00
- Run root: `runs/20260323_091235_ratio_sweep_ratio_30`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `thread_transformer_baseline`，MAE=0.08146608620882034, RMSE=0.17063957452774048
- 与上一轮相比: 上一轮 `import_20260323_091059_main_plus_topconf` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 -0.0237。
- `affect_state_forecaster`: status=completed, MAE=0.09426427632570267, RMSE=0.1650024801492691, Pearson=-0.1983926654635447, Spearman=0.09771581432571362
- `patchtst_baseline`: status=completed, MAE=0.08838911354541779, RMSE=0.1645345687866211, Pearson=-0.02009435294814977, Spearman=0.2961390935637022
- `structure_baseline`: status=completed, MAE=0.12271036207675934, RMSE=0.18300072848796844, Pearson=-0.10856727441298147, Spearman=0.024898860577698575
- `temporal_baseline`: status=completed, MAE=0.13755327463150024, RMSE=0.2121802270412445, Pearson=-0.1619618866757471, Spearman=-0.12963636887411004
- `text_baseline`: status=completed, MAE=0.13152797520160675, RMSE=0.21437649428844452, Pearson=-0.23682333860760996, Spearman=-0.041740989747980885
- `thread_transformer_baseline`: status=completed, MAE=0.08146608620882034, RMSE=0.17063957452774048, Pearson=-0.0393606695984401, Spearman=0.25129052903624755

## 图表解读
- Model metrics: `experiments/figures/import_20260323_091235_ratio_sweep_ratio_30_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `thread_transformer_baseline`。
- Event errors: `experiments/figures/import_20260323_091235_ratio_sweep_ratio_30_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260323_091235_ratio_sweep_ratio_30_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 thread_transformer_baseline，相对文本基线 MAE 改进 0.0501。
- 最佳模型相关系数表现 Pearson=-0.0393606695984401, Spearman=0.25129052903624755。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 支持
- Affect-State Forecaster: MAE=0.09426427632570267, RMSE=0.1650024801492691。
- Text baseline: MAE=0.13152797520160675, RMSE=0.21437649428844452。
- Structure baseline: MAE=0.12271036207675934, RMSE=0.18300072848796844。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 部分支持
- temporal_baseline 相对文本基线的 MAE 改进为 -0.0060。
- structure_baseline 相对文本基线的 MAE 改进为 0.0088。
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
### temporal_baseline 在不同事件上的 MAE 波动较大。
- Possible cause: 跨事件泛化仍不稳定，数据或标签可能受事件主题影响。
- idea.md conflict: 提示当前证据对 event-level 泛化的支撑还有限。
- Recommendation: 后续优先补 cross-event 分析，不建议仅凭当前结果扩展主张。
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
