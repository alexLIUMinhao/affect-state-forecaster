# Experiment Report: 20260319_173108_cross_event_putinmissing

## 实验背景
- 实验类型: 主实验
- 目标假设: H1, H2, H3
- 本轮问题: 验证 held-out event=putinmissing 时各模型的 cross-event 泛化能力。
- 对应 idea.md: 三、核心研究问题, 六、方法方案, 七、实验设计

## 实验任务单
- run_id: `20260319_173108_cross_event_putinmissing`
- 成功标准: affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。
- 失败标准: 若 held-out event 掉点显著，则后续优先做泛化与正则化。
- 数据集: /tmp/pheme_cross_event_nzz5m343/putinmissing_train.jsonl
- ratio: putinmissing_train
- 模型: affect_state_forecaster, structure_baseline, temporal_baseline, text_baseline
- 关键参数: epochs=5, batch_size=16, device=cuda
- 特殊设置: heldout_event=putinmissing

## 运行过程摘要
- 创建时间: 2026-03-19T09:31:41.160344+00:00
- Run root: `runs/20260319_173108_cross_event_putinmissing`
- 日志路径: `experiments/logs/20260319_173108_cross_event_putinmissing.log`
- 总耗时(秒): 0.0
- Warning 数: 0
- Error 数: 0
- 是否全部成功: 是
- 执行命令数: 8

## 结果总览
- 最优模型: `text_baseline`，MAE=0.0964931771159172, RMSE=0.12905849516391754
- 与上一轮相比: 上一轮 `20260319_173034_cross_event_prince_toronto` 的最优模型是 `structure_baseline`，本轮最佳 MAE 变化 0.0091。
- `affect_state_forecaster`: status=completed, MAE=0.10705672204494476, RMSE=0.12132436037063599, Pearson=-0.6656603585936122, Spearman=-0.7074749701108396
- `structure_baseline`: status=completed, MAE=0.10525719076395035, RMSE=0.12649834156036377, Pearson=-0.41196692822341535, Spearman=-0.5020790110464023
- `temporal_baseline`: status=completed, MAE=0.101011723279953, RMSE=0.12950889766216278, Pearson=-0.09215715438447634, Spearman=-0.13693063937629152
- `text_baseline`: status=completed, MAE=0.0964931771159172, RMSE=0.12905849516391754, Pearson=0.2884585138146513, Spearman=0.296683051981965

## 图表解读
- Model metrics: `experiments/figures/20260319_173108_cross_event_putinmissing_model_metrics.png`
  该图用于比较模型整体误差，当前 MAE 最优模型是 `text_baseline`。
- Event errors: `experiments/figures/20260319_173108_cross_event_putinmissing_event_errors.png`
  该图用于查看不同事件上的误差波动，帮助判断 cross-event 稳定性。
- Ratio trends: `experiments/figures/20260319_173108_cross_event_putinmissing_ratio_trends.png`
  当前只有一个 observation ratio，因此该图主要起到占位作用，尚不能判断趋势。

## 对 idea.md 的判断
### H1 早期观测足以预测未来群体情绪
- Verdict: 部分支持
- 最佳模型为 text_baseline，相对文本基线 MAE 改进 0.0000。
- 最佳模型相关系数表现 Pearson=0.2884585138146513, Spearman=0.296683051981965。
### H2 Affect-State Forecaster 优于直接预测基线
- Verdict: 不支持
- Affect-State Forecaster: MAE=0.10705672204494476, RMSE=0.12132436037063599。
- Text baseline: MAE=0.0964931771159172, RMSE=0.12905849516391754。
- Structure baseline: MAE=0.10525719076395035, RMSE=0.12649834156036377。
### H3 时间或结构信息相对文本基线有增益
- Verdict: 不支持
- temporal_baseline 相对文本基线的 MAE 改进为 -0.0045。
- structure_baseline 相对文本基线的 MAE 改进为 -0.0088。
### H4 不同 observation ratio 呈现可解释趋势
- Verdict: 证据不足
- 当前只有 1 个 observation ratio，尚不足以判断趋势。

## 异常现象
### temporal_baseline 未优于 text_baseline。
- Possible cause: 当前时序编码没有有效利用 reply 顺序，或 observation window 信息密度不足。
- idea.md conflict: 削弱了 `idea.md` 中时间动态应带来增益的预期。
- Recommendation: 优先检查时间编码与序列长度处理；单次出现不建议新建 `idea.md`。

## 下一步决策
- 决策动作: 继续，但先做诊断
- 原因: Affect-State 主线被当前结果削弱，需要先确认是实现问题还是假设问题。
- 因此下一步是 优先做 affect-state 消融、正则化和去结构对比。，而不是直接扩大实验范围。
