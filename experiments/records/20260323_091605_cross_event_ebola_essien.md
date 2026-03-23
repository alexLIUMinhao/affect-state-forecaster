# Experiment Report: 20260323_091605_cross_event_ebola_essien

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=ebola-essien 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260323_091605_cross_event_ebola_essien`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_8j60zwfh/ebola-essien_train.jsonl
- ratio: ebola-essien_train
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=ebola-essien

## 运行过程摘要
- 创建时间: 2026-03-23T01:17:02.661779+00:00
- Run root: `runs/20260323_091605_cross_event_ebola_essien`
- 日志路径: `experiments/logs/20260323_091605_cross_event_ebola_essien.log`
- 总耗时(秒): 0.0
- Warning 数: 8
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 12

## 结果总览
- 最优模型: `temporal_baseline`，MAE=0.05972917005419731, RMSE=0.06984865665435791
- 与上一轮相比: 上一轮 `20260323_091509_cross_event_charliehebdo` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 -0.0796。
- `affect_state_forecaster`: status=completed, MAE=0.10613307356834412, RMSE=0.13693884015083313, Pearson=1.0, Spearman=0.9999999999999999
- `patchtst_baseline`: status=completed, MAE=0.09205478429794312, RMSE=0.10595360398292542, Pearson=-1.0, Spearman=-0.9999999999999999
- `structure_baseline`: status=completed, MAE=0.09576579928398132, RMSE=0.11622310429811478, Pearson=-1.0, Spearman=-0.9999999999999999
- `temporal_baseline`: status=completed, MAE=0.05972917005419731, RMSE=0.06984865665435791, Pearson=1.0, Spearman=0.9999999999999999
- `text_baseline`: status=completed, MAE=0.31166332960128784, RMSE=0.341633141040802, Pearson=-1.0, Spearman=-0.9999999999999999
- `thread_transformer_baseline`: status=completed, MAE=0.09379709511995316, RMSE=0.11660042405128479, Pearson=-1.0, Spearman=-0.9999999999999999

## 图表解读
- Model metrics: `experiments/figures/20260323_091605_cross_event_ebola_essien_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `temporal_baseline`。
- Event errors: `experiments/figures/20260323_091605_cross_event_ebola_essien_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260323_091605_cross_event_ebola_essien_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 temporal_baseline，相对文本基线 MAE 改进 0.2519。
- 最佳模型相关系数表现 Pearson=1.0, Spearman=0.9999999999999999。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 部分支持
- Affect-State Forecaster: MAE=0.10613307356834412, RMSE=0.13693884015083313。
- Text baseline: MAE=0.31166332960128784, RMSE=0.341633141040802。
- Structure baseline: MAE=0.09576579928398132, RMSE=0.11622310429811478。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.2519。
- structure_baseline 相对文本基线的 MAE 改进为 0.2159。
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
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
