# Experiment Report: import_20260323_091605_cross_event_ebola_essien

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260323_091605_cross_event_ebola_essien`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260323_091605_cross_event_ebola_essien`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260323_091605_cross_event_ebola_essien
- ratio: unknown
- 模型: affect_state_forecaster, patchtst_baseline, structure_baseline, temporal_baseline, text_baseline, thread_transformer_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-23T01:28:01.846561+00:00
- Run root: `runs/20260323_091605_cross_event_ebola_essien`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `temporal_baseline`，MAE=0.05972917005419731, RMSE=0.06984865665435791
- 与上一轮相比: 上一轮 `import_20260323_091509_cross_event_charliehebdo` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 -0.0796。
- `affect_state_forecaster`: status=completed, MAE=0.10613307356834412, RMSE=0.13693884015083313, Pearson=1.0, Spearman=0.9999999999999999
- `patchtst_baseline`: status=completed, MAE=0.09205478429794312, RMSE=0.10595360398292542, Pearson=-1.0, Spearman=-0.9999999999999999
- `structure_baseline`: status=completed, MAE=0.09576579928398132, RMSE=0.11622310429811478, Pearson=-1.0, Spearman=-0.9999999999999999
- `temporal_baseline`: status=completed, MAE=0.05972917005419731, RMSE=0.06984865665435791, Pearson=1.0, Spearman=0.9999999999999999
- `text_baseline`: status=completed, MAE=0.31166332960128784, RMSE=0.341633141040802, Pearson=-1.0, Spearman=-0.9999999999999999
- `thread_transformer_baseline`: status=completed, MAE=0.09379709511995316, RMSE=0.11660042405128479, Pearson=-1.0, Spearman=-0.9999999999999999

## 图表解读
- Model metrics: `experiments/figures/import_20260323_091605_cross_event_ebola_essien_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `temporal_baseline`。
- Event errors: `experiments/figures/import_20260323_091605_cross_event_ebola_essien_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260323_091605_cross_event_ebola_essien_ratio_trends.png`
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
