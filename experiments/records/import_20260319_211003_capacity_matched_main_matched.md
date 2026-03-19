# Experiment Report: import_20260319_211003_capacity_matched_main_matched

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260319_211003_capacity_matched_main_matched`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260319_211003_capacity_matched_main_matched`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260319_211003_capacity_matched_main_matched
- ratio: unknown
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T13:11:30.362584+00:00
- Run root: `runs/20260319_211003_capacity_matched_main_matched`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `structure_baseline`，MAE=0.13394805788993835, RMSE=0.19480757415294647
- 与上一轮相比: 上一轮 `import_20260319_211003_capacity_matched_main_default` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 0.0166。
- `affect_state_forecaster`: status=completed, MAE=0.1463845670223236, RMSE=0.18740157783031464, Pearson=-0.20935161964771512, Spearman=-0.08935591829709844
- `structure_baseline`: status=completed, MAE=0.13394805788993835, RMSE=0.19480757415294647, Pearson=-0.24220471961087003, Spearman=0.07177318056303753
- `temporal_baseline`: status=completed, MAE=0.17618349194526672, RMSE=0.23209479451179504, Pearson=-0.14436275591914685, Spearman=-0.02886239783991164
- `text_baseline`: status=completed, MAE=0.15234895050525665, RMSE=0.22426491975784302, Pearson=-0.19935484230413142, Spearman=-0.11695413955056115

## 图表解读
- Model metrics: `experiments/figures/import_20260319_211003_capacity_matched_main_matched_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `structure_baseline`。
- Event errors: `experiments/figures/import_20260319_211003_capacity_matched_main_matched_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260319_211003_capacity_matched_main_matched_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 structure_baseline，相对文本基线 MAE 改进 0.0184。
- 最佳模型相关系数表现 Pearson=-0.24220471961087003, Spearman=0.07177318056303753。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 不支持
- Affect-State Forecaster: MAE=0.1463845670223236, RMSE=0.18740157783031464。
- Text baseline: MAE=0.15234895050525665, RMSE=0.22426491975784302。
- Structure baseline: MAE=0.13394805788993835, RMSE=0.19480757415294647。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 -0.0238。
- structure_baseline 相对文本基线的 MAE 改进为 0.0184。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### Affect-State Forecaster 的 MAE 明显差于 structure_baseline。
- Possible cause: 当前显式 affect-state 表达未带来比结构特征更稳定的增益，或模型容量不足。
- idea.md conflict: 与 `idea.md` 中“显式情绪状态建模优于直接预测”主线存在张力。
- Recommendation: 先不新建 `idea.md`；继续做 affect-state 消融与正则化实验。若多轮复现同结论，再考虑新增结构优先版研究计划。 
### temporal_baseline 未优于 text_baseline。
- Possible cause: 当前时序编码没有有效利用 reply 顺序，或 observation window 信息密度不足。
- idea.md conflict: 削弱了 `idea.md` 中时间动态应带来增益的预期。
- Recommendation: 优先检查时间编码与序列长度处理；单次出现不建议新建 `idea.md`。
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
- 原因: Affect-State 主线被当前结果削弱，需要先确认是实现问题还是假设问题。
- 因此下一步是 优先做 affect-state 消融、正则化和去结构对比。，而不是直接扩大实验范围。
