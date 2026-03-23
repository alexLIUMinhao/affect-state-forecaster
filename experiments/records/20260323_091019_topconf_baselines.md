# Experiment Report: 20260323_091019_topconf_baselines

## 实验背景
- 实验类型: 顶会Baseline对比
- 目标假设: H1, H2, H3
- 本轮问题: 顶会可适配baseline在PHEME forecasting设定下是否优于现有自研基础线。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260323_091019_topconf_baselines`
- 成功标准: 至少一个顶会baseline在主要回归指标上进入前二，并可稳定复现。
- 失败标准: 若新增baseline全部落后于现有强基线，则停止继续扩展。
- 数据集: /home/alexmhliu/affect-state-forecaster/data/processed/pheme_forecast_ratio_05_train.jsonl
- ratio: ratio_05_train
- 模型: patchtst_baseline, thread_transformer_baseline, timesnet_baseline
- 关键参数: epochs=1, batch_size=8, device=cuda
- 特殊设置: baseline_family=topconf

## 运行过程摘要
- 创建时间: 2026-03-23T01:10:44.429737+00:00
- Run root: `runs/20260323_091019_topconf_baselines`
- 日志路径: `experiments/logs/20260323_091019_topconf_baselines.log`
- 总耗时(秒): 0.0
- Warning 数: 5
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 6

## 结果总览
- 最优模型: `timesnet_baseline`，MAE=0.10797085613012314, RMSE=0.18161989748477936
- 与上一轮相比: 上一轮 `import_first_round_ratio_05` 的最优模型是 `structure_baseline`，本轮最佳 MAE 变化 -0.0091。
- `patchtst_baseline`: status=completed, MAE=0.11854799836874008, RMSE=0.17262616753578186, Pearson=-0.032635729522494564, Spearman=0.285753341109831
- `thread_transformer_baseline`: status=completed, MAE=0.31523823738098145, RMSE=0.33385637402534485, Pearson=-0.05826493328736684, Spearman=-0.10039520679848353
- `timesnet_baseline`: status=completed, MAE=0.10797085613012314, RMSE=0.18161989748477936, Pearson=-0.11806792748788533, Spearman=0.10090472493485854

## 图表解读
- Model metrics: `experiments/figures/20260323_091019_topconf_baselines_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `timesnet_baseline`。
- Event errors: `experiments/figures/20260323_091019_topconf_baselines_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260323_091019_topconf_baselines_ratio_trends.png`
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
### patchtst_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### thread_transformer_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。
### timesnet_baseline 的 Accuracy 远高于 Macro-F1。
- Possible cause: 多数类占优，分类表现可能被类别不平衡掩盖。
- idea.md conflict: 说明扩展任务分类结果暂时不适合作为强支撑证据。
- Recommendation: 在报告中弱化分类结论，优先以回归指标论证主任务。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
