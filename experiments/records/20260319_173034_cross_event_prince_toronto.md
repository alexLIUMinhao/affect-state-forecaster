# Experiment Report: 20260319_173034_cross_event_prince_toronto

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=prince-toronto 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260319_173034_cross_event_prince_toronto`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_nzz5m343/prince-toronto_train.jsonl
- ratio: prince-toronto_train
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=prince-toronto

## 运行过程摘要
- 创建时间: 2026-03-19T09:31:07.285027+00:00
- Run root: `runs/20260319_173034_cross_event_prince_toronto`
- 日志路径: `experiments/logs/20260319_173034_cross_event_prince_toronto.log`
- 总耗时(秒): 0.0
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 8

## 结果总览
- 最优模型: `structure_baseline`，MAE=0.08739369362592697, RMSE=0.096623994410038
- 与上一轮相比: 上一轮 `20260319_173001_cross_event_ottawashooting` 的最优模型是 `temporal_baseline`，本轮最佳 MAE 变化 -0.041。
- `affect_state_forecaster`: status=completed, MAE=0.14235854148864746, RMSE=0.14334167540073395, Pearson=0.2315764650876369, Spearman=0.30569920271880274
- `structure_baseline`: status=completed, MAE=0.08739369362592697, RMSE=0.096623994410038, Pearson=0.13209289675911634, Spearman=0.04367131467411468
- `temporal_baseline`: status=completed, MAE=0.12840251624584198, RMSE=0.15657365322113037, Pearson=-0.22471565755756967, Spearman=-0.30569920271880274
- `text_baseline`: status=completed, MAE=0.1315266638994217, RMSE=0.15865999460220337, Pearson=0.24491572799268524, Spearman=0.30569920271880274

## 图表解读
- Model metrics: `experiments/figures/20260319_173034_cross_event_prince_toronto_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `structure_baseline`。
- Event errors: `experiments/figures/20260319_173034_cross_event_prince_toronto_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260319_173034_cross_event_prince_toronto_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 structure_baseline，相对文本基线 MAE 改进 0.0441。
- 最佳模型相关系数表现 Pearson=0.13209289675911634, Spearman=0.04367131467411468。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 不支持
- Affect-State Forecaster: MAE=0.14235854148864746, RMSE=0.14334167540073395。
- Text baseline: MAE=0.1315266638994217, RMSE=0.15865999460220337。
- Structure baseline: MAE=0.08739369362592697, RMSE=0.096623994410038。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0031。
- structure_baseline 相对文本基线的 MAE 改进为 0.0441。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### Affect-State Forecaster 的 MAE 明显差于 structure_baseline。
- Possible cause: 当前显式 affect-state 表达未带来比结构特征更稳定的增益，或模型容量不足。
- idea.md conflict: 与 `idea.md` 中“显式情绪状态建模优于直接预测”主线存在张力。
- Recommendation: 先不新建 `idea.md`；继续做 affect-state 消融与正则化实验。若多轮复现同结论，再考虑新增结构优先版研究计划。 

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: Affect-State 主线被当前结果削弱，需要先确认是实现问题还是假设问题。
- 因此下一步是 优先做 affect-state 消融、正则化和去结构对比。，而不是直接扩大实验范围。
