# Experiment Report: 20260319_173142_cross_event_sydneysiege

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=sydneysiege 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260319_173142_cross_event_sydneysiege`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_nzz5m343/sydneysiege_train.jsonl
- ratio: sydneysiege_train
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=sydneysiege

## 运行过程摘要
- 创建时间: 2026-03-19T09:32:14.805959+00:00
- Run root: `runs/20260319_173142_cross_event_sydneysiege`
- 日志路径: `experiments/logs/20260319_173142_cross_event_sydneysiege.log`
- 总耗时(秒): 0.0
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 8

## 结果总览
- 最优模型: `affect_state_forecaster`，MAE=0.11327497661113739, RMSE=0.18678252398967743
- 与上一轮相比: 上一轮 `20260319_173108_cross_event_putinmissing` 的最优模型是 `text_baseline`，本轮最佳 MAE 变化 0.0168。
- `affect_state_forecaster`: status=completed, MAE=0.11327497661113739, RMSE=0.18678252398967743, Pearson=-0.1044950089866099, Spearman=-0.1474535960011952
- `structure_baseline`: status=completed, MAE=0.12005847692489624, RMSE=0.20020215213298798, Pearson=-0.430304445302535, Spearman=-0.015748369753641706
- `temporal_baseline`: status=completed, MAE=0.1368206888437271, RMSE=0.220431849360466, Pearson=-0.18454280134806825, Spearman=-0.040916152203644904
- `text_baseline`: status=completed, MAE=0.1754033863544464, RMSE=0.2425088882446289, Pearson=-0.19995166408773965, Spearman=-0.04996545045221223

## 图表解读
- Model metrics: `experiments/figures/20260319_173142_cross_event_sydneysiege_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `affect_state_forecaster`。
- Event errors: `experiments/figures/20260319_173142_cross_event_sydneysiege_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260319_173142_cross_event_sydneysiege_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 部分支持
- 最佳模型为 affect_state_forecaster，相对文本基线 MAE 改进 0.0621。
- 最佳模型相关系数表现 Pearson=-0.1044950089866099, Spearman=-0.1474535960011952。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 支持
- Affect-State Forecaster: MAE=0.11327497661113739, RMSE=0.18678252398967743。
- Text baseline: MAE=0.1754033863544464, RMSE=0.2425088882446289。
- Structure baseline: MAE=0.12005847692489624, RMSE=0.20020215213298798。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0386。
- structure_baseline 相对文本基线的 MAE 改进为 0.0553。
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
