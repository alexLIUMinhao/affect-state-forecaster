# Experiment Report: import_20260319_174402_labeler_lexicon_conservative_ratio_05

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260319_174402_labeler_lexicon_conservative_ratio_05`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260319_174402_labeler_lexicon_conservative_ratio_05`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260319_174402_labeler_lexicon_conservative_ratio_05
- ratio: unknown
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T09:44:46.439773+00:00
- Run root: `runs/20260319_174402_labeler_lexicon_conservative_ratio_05`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 0

## 结果总览
- 最优模型: `affect_state_forecaster`，MAE=0.0386434830725193, RMSE=0.04187716916203499
- 与上一轮相比: 上一轮 `import_20260319_174329_labeler_lexicon_v1_ratio_05` 的最优模型是 `affect_state_forecaster`，本轮最佳 MAE 变化 -0.0787。
- `affect_state_forecaster`: status=completed, MAE=0.0386434830725193, RMSE=0.04187716916203499, Pearson=0.13934132209423658, Spearman=0.16737642679484713
- `structure_baseline`: status=completed, MAE=0.045242927968502045, RMSE=0.05790647491812706, Pearson=-0.11138661316718404, Spearman=-0.11316557409033681
- `temporal_baseline`: status=completed, MAE=0.05112168565392494, RMSE=0.06512842327356339, Pearson=-0.03049118831547202, Spearman=-0.04785775648465584
- `text_baseline`: status=completed, MAE=0.09661539644002914, RMSE=0.1254686713218689, Pearson=-0.11820610048985879, Spearman=-0.04861208189334569

## 图表解读
- Model metrics: `experiments/figures/import_20260319_174402_labeler_lexicon_conservative_ratio_05_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `affect_state_forecaster`。
- Event errors: `experiments/figures/import_20260319_174402_labeler_lexicon_conservative_ratio_05_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260319_174402_labeler_lexicon_conservative_ratio_05_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 支持
- 最佳模型为 affect_state_forecaster，相对文本基线 MAE 改进 0.0580。
- 最佳模型相关系数表现 Pearson=0.13934132209423658, Spearman=0.16737642679484713。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 支持
- Affect-State Forecaster: MAE=0.0386434830725193, RMSE=0.04187716916203499。
- Text baseline: MAE=0.09661539644002914, RMSE=0.1254686713218689。
- Structure baseline: MAE=0.045242927968502045, RMSE=0.05790647491812706。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 支持
- temporal_baseline 相对文本基线的 MAE 改进为 0.0455。
- structure_baseline 相对文本基线的 MAE 改进为 0.0514。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### temporal_baseline 的 Pearson/Spearman 接近 0。
- Possible cause: 模型可能只学到了均值附近回归，未捕捉 thread 级差异。
- idea.md conflict: 削弱了“早期观测足以预测未来走势”的论点。
- Recommendation: 需要检查标签噪声、事件切分与模型表达；若多轮如此，再考虑修订研究主线。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: 当前结果可继续推进，但仍有未解释的异常现象需要优先澄清。
- 因此下一步是 围绕报告中的异常项安排定向诊断实验，而不是直接扩大模型复杂度。，而不是直接扩大实验范围。
