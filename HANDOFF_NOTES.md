# RelTwin v36 合作者交接

底稿：GitHub `paper` 的 `b28b3e9`，经作者确认优先于附件中的 PDF 文件名。本轮未修改 Experiments、Results、Conclusion 或两张表。本文件中的英文为建议替换文本，尚未合入论文。

## 1. Experimental setup：建议重组，不删复现信息

建议将 3.1 按三个自然段组织，去掉连续加粗的段首标签：

- 数据集与样本构造：保留三个公开基准规模、ESC-50 类别与音频划分、模板、关系/类别查询数量、音源隔离；保留静音干预的音频重组参数及 40 条实际发生变化的子集。
- 评价与统计：保留 set-IoU、R1、PairAcc、JointPairAcc、Swap error、same-answer 的定义，空并集、并列规则，20,000 次分组 bootstrap、39 个连通组及最大组 18 条，三种子均值与样本标准差。
- 参数、训练与推理：保留两个训练阶段、LoRA/连接层、权重、温度、replay 配比、候选构造、机制对照匹配条件、梯度计算、种子及生成设置，不为了凑段落删掉现有参数。

沿用的协议就近引用相应来源；本文自行设定的超参照实陈述。没有原始记录支持，不增写 “following prior work” 或 “selected on the development set”。

表 1 SpotSound-A + RelTwin：作者此前确认来自同事的完整配置实验，该确认保留；对应 checkpoint、两阶段训练设置与逐行预测仍需按 P0-10 归档。不能因为它与某个候选监督结果数值相同就认定来自同一次实验。

## 2. 候选监督消融：现有 3.3 已独立成节

建议标题：

```latex
\subsection{Ablation: contribution of candidate supervision}
```

建议在本节开头用以下句子交代职责，保留后续结果及置信区间：

> This matched Stage-1 ablation on SpotSound-A compares paired supervised fine-tuning with and without candidate supervision, holding the initialization, training data, updates, and inference settings fixed. Table 2 isolates the contribution of $\mathcal L_{\rm cand}$ to localization and query-specific selection.

此表不是 JS、Stage 2 和 replay 的完整组件消融。无需为了本轮改稿补造数据或新增实验。

## 3. Conclusion：供合作者确认的英文替换段

> RelTwin improves two backbones across three public grounding benchmarks, with the adapted SpotSound-A achieving the strongest results among the methods listed in Table 1. Public evaluations establish broad localization gains, while matched inverse-query tests support query-specific selection between real event occurrences beyond recognizing sound categories. Candidate supervision retains its advantage under the tested changes to silence timing, providing complementary evidence of selection across altered temporal layouts. The framework extends answer competition to interval-set quality through a second learning stage and preserves direct timestamp generation at inference.

本段对 Stage 2 描述的是设计范围，不声称独立验证每项贡献；对静音干预限定为已测变化，不声称完全排除位置捷径。

3.4 末段仍有受保护的 `position-independent relation learning`。建议仅将相关句子改为：

> Together, public benchmarks test localization transfer, paired tests isolate query-specific selection, and timing interventions examine whether its advantage persists under altered silence layouts; set-level learning extends answer comparison to interval coverage and boundaries.

## 4. 数值核查：先查原始精度，不直接改数

Table 2 显示 59.43 与 58.42，显示值之差为 1.01，而增益行和正文写 1.02。请提供未舍入的原始分数及增益计算记录，核对是否同一评测与样本集合；目前无法认定差异来自四舍五入。本轮保持表格、正文数字和置信区间不变。

## 5. 作者与提交前确认

- 五位作者顺序、邮箱、共同一作和 Wei Xu / Zheng Zhang 双通讯均保留当前稿。尽管此前会话已确认双通讯，仍请全体作者在提交前确认资格及标记；本轮未自行调整。
- PDF 中五位作者姓名分别链接到本人 ORCID，已检查链接区域与 URI 对应。ORCID 数字没有额外排成可见文字行；不将其描述为可见的五串号码。
- 尚未完成 ORCID 外部身份核验或投稿系统绑定；这些不能由排版检查替代。
- 本轮无新增实验、外部出版元数据复核或投稿门户规则复核，不将本稿称为投稿终稿。
