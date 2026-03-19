# Experiment Report: import_20260319_174149_labeler_lexicon_v1_ratio_50

## 实验背景
- 实验类型: 历史结果导入
- 目标假设: H1, H2, H3, H4
- 本轮问题: 回填历史实验 `20260319_174149_labeler_lexicon_v1_ratio_50`，确认其对当前研究主线的支持程度。
- 对应 idea.md: 三、核心研究问题, 七、实验设计

## 实验任务单
- run_id: `import_20260319_174149_labeler_lexicon_v1_ratio_50`
- 成功标准: 能够生成完整报告，并对 idea.md 给出可读判断。
- 失败标准: 若历史结果缺失严重，则仅标记证据不足，不给过度结论。
- 数据集: runs/20260319_174149_labeler_lexicon_v1_ratio_50
- ratio: unknown
- 模型: text_baseline
- 关键参数: epochs=unknown, batch_size=unknown, device=unknown
- 特殊设置: 来自既有 runs 目录的历史导入

## 运行过程摘要
- 创建时间: 2026-03-19T09:44:45.647677+00:00
- Run root: `runs/20260319_174149_labeler_lexicon_v1_ratio_50`
- 日志路径: `none`
- 总耗时(秒): None
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 否
- 执行命令数: 0

## 结果总览
- 与上一轮相比: 上一轮 `import_20260319_173355_labeler_lexicon_v1_ratio_50` 的最优模型是 ``，本轮最佳 MAE 变化 None。
- `text_baseline`: status=failed, MAE=None, RMSE=None, Pearson=None, Spearman=None

## 图表解读
- Model metrics: `experiments/figures/import_20260319_174149_labeler_lexicon_v1_ratio_50_model_metrics.png`
  该图用于比较模型整体误差。
- Event errors: `experiments/figures/import_20260319_174149_labeler_lexicon_v1_ratio_50_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/import_20260319_174149_labeler_lexicon_v1_ratio_50_ratio_trends.png`
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
- 缺少时间或结构基线结果。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 0 个 observation ratio，尚不足以判断趋势。

## 异常现象
- No material mismatch signals detected in this run.

## 下一步决策
- 决策动作: 暂停方法扩展，先修复工程问题
- 原因: 存在失败模型：text_baseline，当前结果不能作为研究结论。
- 因此下一步是 修复失败模型后重新运行同一轮实验。，而不是直接扩大实验范围。
