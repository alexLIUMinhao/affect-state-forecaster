# Experiment Report: 20260323_091703_cross_event_ferguson

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=ferguson 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260323_091703_cross_event_ferguson`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_8j60zwfh/ferguson_train.jsonl
- ratio: ferguson_train
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=ferguson

## 运行过程摘要
- 创建时间: 2026-03-23T01:17:58.401632+00:00
- Run root: `runs/20260323_091703_cross_event_ferguson`
- 日志路径: `experiments/logs/20260323_091703_cross_event_ferguson.log`
- 总耗时(秒): 0.0
- Warning 数: 8
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 12

## 结果总览
- 最优模型: `thread_transformer_baseline`，MAE=0.10716671496629715, RMSE=0.1362522393465042
- 与上一轮相比: 上一轮 `20260323_091605_cross_event_ebola_essien` 的最优模型是 `temporal_baseline`，本轮最佳 MAE 变化 0.0474。
- `affect_state_forecaster`: status=completed, MAE=0.10926328599452972, RMSE=0.12777908146381378, Pearson=0.21548259900590427, Spearman=0.27684423330767427
- `patchtst_baseline`: status=completed, MAE=0.10761513561010361, RMSE=0.12585605680942535, Pearson=0.30333235529087227, Spearman=0.3618000617643046
- `structure_baseline`: status=completed, MAE=0.11619963496923447, RMSE=0.13751043379306793, Pearson=-0.0014188351253872436, Spearman=0.02643488823373347
- `temporal_baseline`: status=completed, MAE=0.11054041236639023, RMSE=0.14183148741722107, Pearson=0.3752521705167183, Spearman=0.3828349490761827
- `text_baseline`: status=completed, MAE=0.1418616771697998, RMSE=0.17566870152950287, Pearson=0.10046740753369063, Spearman=0.09544187675837262
- `thread_transformer_baseline`: status=completed, MAE=0.10716671496629715, RMSE=0.1362522393465042, Pearson=0.21327350226247807, Spearman=0.26692330138147496

## 图表解读
- Model metrics: `experiments/figures/20260323_091703_cross_event_ferguson_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `thread_transformer_baseline`。
- Event errors: `experiments/figures/20260323_091703_cross_event_ferguson_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260323_091703_cross_event_ferguson_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 thread_transformer_baseline，相对文本基线 MAE 改进 0.0347。
- 最佳模型相关系数表现 Pearson=0.21327350226247807, Spearman=0.26692330138147496。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 支持
- Affect-State Forecaster: MAE=0.10926328599452972, RMSE=0.12777908146381378。
- Text baseline: MAE=0.1418616771697998, RMSE=0.17566870152950287。
- Structure baseline: MAE=0.11619963496923447, RMSE=0.13751043379306793。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0313。
- structure_baseline 相对文本基线的 MAE 改进为 0.0257。
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
