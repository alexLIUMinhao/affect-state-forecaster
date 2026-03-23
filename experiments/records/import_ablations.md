# Experiment Report: import_ablations

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `ablations`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_ablations`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/ablations
- ratio: unknown
- 模型: asf_full, asf_no_affect_state, asf_no_structure, asf_no_temporal, asf_replies_only, asf_source_only
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T09:44:46.998584+00:00
- Run root: `runs/ablations`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `asf_no_affect_state`，MAE=0.10793595016002655, RMSE=0.17799146473407745
- 与上一轮相比: 上一轮 `import_20260319_174402_labeler_lexicon_conservative_ratio_05` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 0.0693。
- `asf_full`: status=completed, MAE=0.11735665053129196, RMSE=0.17294029891490936, Pearson=-0.04726908080453998, Spearman=0.19641030880448201
- `asf_no_affect_state`: status=completed, MAE=0.10793595016002655, RMSE=0.17799146473407745, Pearson=-0.0496919254465246, Spearman=0.2769169909964379
- `asf_no_structure`: status=completed, MAE=0.11986560374498367, RMSE=0.17837591469287872, Pearson=-0.05239417847449138, Spearman=-0.05080743506242312
- `asf_no_temporal`: status=completed, MAE=0.11860393732786179, RMSE=0.1742217093706131, Pearson=-0.12358072500104328, Spearman=0.09265880219549673
- `asf_replies_only`: status=completed, MAE=0.10795808583498001, RMSE=0.17718175053596497, Pearson=-0.12649215922818927, Spearman=0.14522896235733457
- `asf_source_only`: status=completed, MAE=0.11656317859888077, RMSE=0.1750260591506958, Pearson=-0.1670978292288296, Spearman=-0.08365822100606099

## 图表解读
- Model metrics: `experiments/figures/import_ablations_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `asf_no_affect_state`。
- Event errors: `experiments/figures/import_ablations_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_ablations_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 证据不足
- 缺少文本基线或总体最佳模型结果。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 证据不足
- 缺少 affect-state 或文本基线结果。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 证据不足
- 缺少文本基线结果。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### asf_full 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### asf_no_affect_state 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### asf_no_structure 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### asf_no_temporal 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### asf_replies_only 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### asf_source_only 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
