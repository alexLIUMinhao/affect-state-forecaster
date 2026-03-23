# Experiment Report: 20260323_092051_cross_event_putinmissing

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=putinmissing 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260323_092051_cross_event_putinmissing`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_8j60zwfh/putinmissing_train.jsonl
- ratio: putinmissing_train
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=putinmissing

## 运行过程摘要
- 创建时间: 2026-03-23T01:21:48.277643+00:00
- Run root: `runs/20260323_092051_cross_event_putinmissing`
- 日志路径: `experiments/logs/20260323_092051_cross_event_putinmissing.log`
- 总耗时(秒): 0.0
- Warning 数: 8
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 12

## 结果总览
- 最优模型: `thread_transformer_baseline`，MAE=0.06605269014835358, RMSE=0.10800120234489441
- 与上一轮相比: 上一轮 `20260323_091953_cross_event_prince_toronto` 的最优模型是 `patchtst_baseline`，本轮最佳 MAE 变化 -0.0201。
- `affect_state_forecaster`: status=completed, MAE=0.11714617162942886, RMSE=0.12756317853927612, Pearson=-0.2778510780670371, Spearman=-0.3423265984407288
- `patchtst_baseline`: status=completed, MAE=0.06807126849889755, RMSE=0.1095673218369484, Pearson=0.09999925321995233, Spearman=0.2520920866827581
- `structure_baseline`: status=completed, MAE=0.10525719821453094, RMSE=0.12649832665920258, Pearson=-0.41196702531005774, Spearman=-0.5020790110464023
- `temporal_baseline`: status=completed, MAE=0.1010117381811142, RMSE=0.12950889766216278, Pearson=-0.09215752898498142, Spearman=-0.13693063937629152
- `text_baseline`: status=completed, MAE=0.0964931920170784, RMSE=0.12905851006507874, Pearson=0.2884584550385917, Spearman=0.296683051981965
- `thread_transformer_baseline`: status=completed, MAE=0.06605269014835358, RMSE=0.10800120234489441, Pearson=0.19949125819557614, Spearman=0.15975241260567344

## 图表解读
- Model metrics: `experiments/figures/20260323_092051_cross_event_putinmissing_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `thread_transformer_baseline`。
- Event errors: `experiments/figures/20260323_092051_cross_event_putinmissing_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260323_092051_cross_event_putinmissing_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 thread_transformer_baseline，相对文本基线 MAE 改进 0.0304。
- 最佳模型相关系数表现 Pearson=0.19949125819557614, Spearman=0.15975241260567344。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 不支持
- Affect-State Forecaster: MAE=0.11714617162942886, RMSE=0.12756317853927612。
- Text baseline: MAE=0.0964931920170784, RMSE=0.12905851006507874。
- Structure baseline: MAE=0.10525719821453094, RMSE=0.12649832665920258。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 不支持
- temporal_baseline 相对文本基线的 MAE 改进为 -0.0045。
- structure_baseline 相对文本基线的 MAE 改进为 -0.0088。
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

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: Affect-State 主线被当前结果削弱，需要先确认是实现问题还是假设问题。
- 因此下一步是 优先做 affect-state 消融、正则化和去结构对比。，而不是直接扩大实验范围。
