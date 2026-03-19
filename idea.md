
## 一、研究题目

**面向公共事件的多模态情绪驱动演化预测研究**
英文可写为：**Multimodal Affect-Driven Event Evolution Forecasting**

## 二、研究背景与问题定位

现有研究大致分成三类。第一类是多模态情感分析与对话情绪识别，例如 MELD 和 M3ED，这类工作强调文本、语音、视觉与上下文对**当前情绪识别**的重要性，但任务目标仍然是识别当前 utterance 或当前片段的情绪标签。MELD 约有 1,433 段对话、约 13,000 条 utterances；M3ED 含 990 段双人对话、9,082 个 turns 和 24,449 条 utterances。([ACL Anthology][2])

第二类是 rumor / veracity / stance / event-thread prediction，这类工作强调事件线程、回复树、时间顺序和传播结构，但主要输出是真假、立场或谣言标签，而不是未来情绪走势。PHEME 是这条线最常见的数据基础之一，原始公开版本是 breaking news 下的 rumor / non-rumor Twitter 线程数据，RumourEval 2019 也沿用了线程式 rumor verification 设定，并在 2017 数据基础上扩展了新的测试数据。([Figshare][3])

第三类是更接近你的方向的 forecasting 工作。ACL 2025 的 *Context-Aware Sentiment Forecasting* 已经明确研究“根据事件发展预测未来 sentiment response”；MM-Forecast 则研究“多模态 temporal event forecasting”，并构造了 MidEast-TE-mm 数据集。说明“未来情绪预测”和“多模态事件预测”都不是空白，但它们还没有自然地闭合成**事件级、群体级、情绪演化预测**这一更具体的任务。([ACL Anthology][1])

因此，这篇论文的问题定位不是“从零发明一个完全没人想过的方向”，而是：

> **在真实公共事件的早期观测阶段，利用文本、传播结构与可扩展的多模态证据，预测事件后续阶段的群体情绪走势。**

这个定位的价值在于：它把“当前情绪识别”推进到“未来情绪演化预测”，把“单用户未来反应”收紧到“事件线程级/群体级未来情绪”，同时保留了事件传播任务中最重要的时间与结构约束。([ACL Anthology][4])

## 三、核心研究问题

这篇论文围绕三个核心问题展开。

第一，**事件早期观测是否足以预测后续群体情绪走势**。也就是：仅看 source post、早期回复、时间顺序与传播结构，能否对未来窗口中的负面情绪比例或情绪分布做出稳定预测。这个问题对应 ACL 2025 那篇里“predict imminent sentiment response to ongoing events”的前瞻性设定。([ACL Anthology][1])

第二，**情绪应不应该被建模为动态状态变量，而不是最终标签**。现有多数方法要么识别当前情绪，要么直接输出未来 sentiment；你的方法核心假设是：情绪是连接“早期观测”和“未来走势”的中间动力状态。([ACL Anthology][4])

第三，**事件线程中的时间顺序与传播结构，对未来情绪预测到底有多大帮助**。这一步承接 rumor / conversation-tree 方向的已有发现，但将其转向 affect forecasting，而不是继续做真假或立场识别。([ACL Anthology][5])

## 四、任务定义

第一篇论文不宜做太大，建议先收成一个**单主任务 + 一个扩展任务**。

主任务定义为：

给定某个事件线程在观察窗口内的早期信息
[
X = {x_1, x_2, \ldots, x_T}
]
其中包含 source text、reply texts、timestamps、reply tree / propagation structure，以及可扩展的图像或外部事件上下文，预测未来窗口中的
[
y = \text{future_neg_ratio}
]
即后续窗口内的**负面情绪比例**。

扩展任务是预测未来窗口内的完整情绪分布：
[
y = [p_{neg}, p_{neu}, p_{pos}]
]

主任务先做回归，扩展任务再做 distribution forecasting。这样最稳，因为 negative ratio 既容易构造标签，也容易和预警场景挂钩。ACL 2025 那篇已经证明“未来 sentiment forecasting”本身具备研究意义；你这里只是把目标从 user-level / response-level 收紧成 thread-level / group-level。([ACL Anthology][4])

## 五、数据与 benchmark 构建方案

### 5.1 主数据集

第一阶段建议以 **PHEME** 为主。原因不是它天然带“未来情绪标签”，而是它天然带**事件、线程、时间、回复树**，非常适合被重构成 forecasting benchmark。PHEME 的公开说明明确指出它包含 breaking news 期间的 rumor / non-rumor 线程；后续 rumor verification 工作也一直把它当作“rumor unfolding as threads”来用。([Figshare][3])

辅助数据可用 **RumourEval 2019** 做外部验证或迁移测试，因为它同样基于 rumor thread，2017 训练语料包含 8 个 breaking-news events、297 个 source tweets 和约 7,100 条 discussion tweets。([ACL Anthology][6])

### 5.2 benchmark 重构协议

你不直接继承原始 rumor / veracity 标签，而是把每个 thread 重构成 observation–forecast 样本。

具体协议如下：

* 按时间排序每个 thread 中的 replies；
* 用前 50% 回复作为 observation window，后 50% 作为 forecast window；
* 再补两个 setting：30/70 和 70/30；
* 对 forecast window 内的回复文本做 sentiment labeling，统计负面、中性、正面比例；
* 生成每个 thread 的 `future_neg_ratio`、`future_neu_ratio`、`future_pos_ratio` 和 `future_majority_sentiment`。

这里最关键的是：**按事件切分 train/val/test，而不是随机打散线程**。RumourEval 和 PHEME 系工作都强调 thread / event 级 generalization 的重要性；如果随机混洗，容易出现信息泄漏，让结果虚高。([ACL Anthology][6])

### 5.3 情绪标签构造

由于 PHEME 原始标签不是情绪标签，第一版可采用**自动情绪标注**。具体做法是：

* 用现成 sentiment model 对 replies 打三分类标签；
* 在一小部分样本上做人审，估算标注误差；
* 把自动标签视为 weak supervision，而非 ground-truth gold annotation。

这一步是方案中的一个风险点，但也是论文的工程可行性基础。只要你诚实说明“这是 benchmark reconstruction，而不是原生 gold benchmark”，这个策略是可以接受的。

## 六、方法方案

### 6.1 总体思路

方法不建议一上来太复杂。第一篇论文的主线应当是：

> **先证明任务成立，再证明“显式情绪状态建模”优于直接预测。**

所以方法结构建议分成四层：

1. **输入编码层**：编码 source text、early replies、时间位置、传播结构；
2. **时序聚合层**：建模观察窗口中的动态变化；
3. **情绪状态层**：显式估计当前群体 affect state；
4. **未来预测层**：基于该状态预测 `future_neg_ratio` 或未来情绪分布。

### 6.2 第一阶段输入模态

为了稳妥，第一版“多模态”先不要强行上图像视频。第一阶段可以这样定义：

* 文本模态：source + replies；
* 时间模态：timestamps / time bins；
* 结构模态：reply tree / parent-child relation。

也就是说，第一阶段的“多模态”可解释为**语义 + 时间 + 结构**的 heterogeneous modalities。这样更稳，也更符合 PHEME 数据现实。等第一篇跑通，再考虑接入真正的视觉证据，向 MM-Forecast 的方向延展。([ACM数字图书馆][7])

### 6.3 基线模型

基线分三层。

第一层，**静态文本基线**。
用 BERT / RoBERTa 对 source + observed replies 拼接后编码，直接回归 `future_neg_ratio`。它回答的是：只靠早期文本语义，不建模时间和结构，能做到什么程度。

第二层，**时序文本基线**。
按时间片组织 observed replies，用 LSTM 或 Temporal Transformer 建模，再输出未来负面比例。它回答的是：时间动态本身有没有增益。

第三层，**结构感知基线**。
把 reply tree 编成 Tree-LSTM、GNN 或 graph pooling 表示，再做预测。它回答的是：传播结构对未来情绪走势是否有用。PHEME / RumourEval 周边工作的一个共识就是 thread structure 很重要。([ACL Anthology][5])

### 6.4 拟提出的方法

你的核心方法可以命名为：

**Affect-State Forecaster (ASF)**

核心设计是：模型不是直接从 early observations 跳到 future target，而是先学习一个 latent group affect state，再用它预测 future negative ratio。

可以写成两步：

* (h_t = f_{\text{enc}}(x_{\le T})) ：对观察窗口建模；
* (z_{aff} = g(h_t)) ：估计当前群体情绪状态；
* (\hat y = q(z_{aff})) ：预测未来负面情绪比例。

这样做的研究假设是：**情绪是未来事件演化的中间驱动状态**。这也是整篇论文最值得强调的方法点。

### 6.5 可选增强

如果第一版结果不够强，可加两个增强但不改主框架：

* **Affect consistency regularization**：当前状态与早期窗口中的平均情绪统计保持一致；
* **Structure-aware temporal fusion**：时间序列表示和传播树表示做 cross-attention 或 gated fusion。

## 七、实验设计

### 7.1 评测指标

主任务为回归，使用：

* MAE
* RMSE
* Pearson correlation
* Spearman correlation

扩展任务若做 future majority sentiment 分类，则增加：

* Accuracy
* Macro-F1
* Weighted-F1

如果做未来情绪分布预测，再增加：

* KL divergence
* Jensen–Shannon divergence

### 7.2 对比设置

必须做三种 observation ratio：

* 30% → 70%
* 50% → 50%
* 70% → 30%

这三种设置对应三类能力：

* 早期预警能力
* 常规预测能力
* 后期修正能力

### 7.3 消融实验

至少做六组：

1. 去掉时间建模；
2. 去掉结构建模；
3. 去掉 affect-state 中间层；
4. 只用 source，不用 replies；
5. 不同 sentiment labeler 的稳健性对比；
6. 不同 observation ratio 的对比。

### 7.4 泛化实验

建议做两个泛化测试：

* **cross-event**：按事件留一法或事件级切分；
* **cross-dataset**：PHEME 上训练，RumourEval 上迁移验证。

这样可以增强论文说服力，因为你研究的是“未来事件情绪预测”，不是只记住固定事件表面模式。([ACL Anthology][6])

## 八、related work 结构

论文 related work 建议分四节。

第一节，**Multimodal Sentiment Analysis / Emotion Recognition**，以 MELD、M3ED 为代表，说明它们关注的是当前情绪识别。([ACL Anthology][2])

第二节，**Rumor / Event-Thread Modeling**，说明 thread structure、time ordering 和 propagation are important，但输出主要是 rumor / stance / veracity。([ACL Anthology][6])

第三节，**Sentiment Forecasting**，重点讨论 ACL 2025 那篇，说明未来 sentiment prediction 已被明确提出，但更偏 user response forecasting。([ACL Anthology][1])

第四节，**Multimodal Temporal Event Forecasting**，讨论 MM-Forecast，说明多模态未来事件预测已被提出，但 affect 还不是核心预测对象。([ACM数字图书馆][7])

## 九、预期创新点

这篇论文的创新点不应写成“首次提出未来情绪预测”，那不成立。更稳的写法是：

第一，提出一个**事件级、群体级**的未来情绪演化预测设定，把 sentiment forecasting 从 user-level response 扩展到 event-thread level evolution。([ACL Anthology][4])

第二，基于 PHEME / RumourEval 重构一个 observation-to-future benchmark，并采用事件级切分与多 observation ratio 评测协议。([ACL Anthology][6])

第三，提出一种显式 affect-state 中间层，将情绪从最终标签转化为动态状态变量。

第四，系统分析语义、时间和结构三类信息在未来情绪演化预测中的作用边界。

## 十、实施计划

### 第 1 阶段：benchmark 构建（2–3 周）

完成 PHEME 数据整理、thread 时间序列重构、情绪弱标注和 train/val/test 切分。输出第一版 benchmark。

### 第 2 阶段：基线复现（2–3 周）

跑通静态文本、时序文本、结构感知三组 baseline，完成基本结果表。

### 第 3 阶段：核心方法实现（2–4 周）

实现 Affect-State Forecaster，完成主实验和消融实验。

### 第 4 阶段：泛化与补充实验（2 周）

做 cross-event、cross-dataset、不同 observation ratio 实验。

### 第 5 阶段：写作与打磨（2–3 周）

完成引言、方法、实验和 case study，补齐 Table 1、Table 2、Figure 1。

## 十一、风险与备选方案

最大风险是：**PHEME 没有原生情绪标签，benchmark 依赖自动标注。**
对应方案是：先把论文贡献重心放在“任务重构 + 评测协议 + 方法框架”，同时在小样本人审上验证标签质量。

第二个风险是：**第一版没有真正视觉模态。**
对应方案是先把“多模态”表述成 heterogeneous modalities（语义、时间、结构），后续若进展顺利，再引入带图像的事件数据，向 MM-Forecast 式设定扩展。([ACM数字图书馆][7])

第三个风险是：**任务过新导致 baseline 不统一。**
对应方案是主动把 baseline 组织成三条线：sentiment forecasting、event-thread modeling、temporal forecasting。这样反而能体现你在“任务标准化”上的贡献。

## 十二、一段可以直接用的方案摘要

本文拟研究面向公共事件的多模态情绪驱动演化预测问题，目标是在事件早期阶段，基于文本内容、时间顺序与传播结构等观测信息，预测事件后续阶段的群体情绪走势。研究首先基于 PHEME 与 RumourEval 等事件线程数据，重构 observation-to-future 的 benchmark，并以未来窗口中的负面情绪比例作为核心预测目标；在此基础上，构建静态文本、时序文本与结构感知等基线模型，进一步提出显式情绪状态建模方法，将群体情绪表示为连接早期观测与未来演化的中间动态状态。实验将采用事件级数据划分、多 observation ratio 设置及回归/分类联合评测，系统分析语义、时间与结构信息在未来情绪演化预测中的作用，并验证情绪状态建模对公共事件早期预警任务的有效性。([ACL Anthology][4])


[1]: https://aclanthology.org/2025.acl-long.136/?utm_source=chatgpt.com "Context-Aware Sentiment Forecasting via LLM-based Multi ..."
[2]: https://aclanthology.org/P19-1050/?utm_source=chatgpt.com "MELD: A Multimodal Multi-Party Dataset for Emotion ..."
[3]: https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619?utm_source=chatgpt.com "PHEME dataset of rumours and non-rumours"
[4]: https://aclanthology.org/2025.acl-long.136.pdf?utm_source=chatgpt.com "Context-Aware Sentiment Forecasting via LLM-based Multi ..."
[5]: https://aclanthology.org/2024.lrec-main.860.pdf?utm_source=chatgpt.com "Knowledge Graphs for Real-World Rumour Verification"
[6]: https://aclanthology.org/S19-2147.pdf?utm_source=chatgpt.com "SemEval-2019 Task 7: RumourEval, Determining Rumour ..."
[7]: https://dl.acm.org/doi/10.1145/3664647.3681593?utm_source=chatgpt.com "MM-Forecast: A Multimodal Approach to Temporal Event ..."
