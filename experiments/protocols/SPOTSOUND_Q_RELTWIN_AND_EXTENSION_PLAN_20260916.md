# SpotSound-Q RelTwin 与跨底座扩展实验计划

## 1. 研究问题与最终目标

本轮实验回答两个递进问题：

1. RelTwin 是否能从 SpotSound-A 迁移到同系列但结构不同的 SpotSound-Q；
2. 若 Q 上成立，RelTwin 是否还能迁移到独立公开模型 TimeAudio。

论文主线仍以 **RelTwin** 为方法名。Q 实验负责证明方法不是只对 SpotSound-A 有效；TimeAudio 负责进一步证明跨模型家族的可迁移性。所有结论以同协议实测为准，不混用无法核准样本清单的跨论文数字。

### 2026-09-16 Q-P0 审计结论

SpotSound-Q 当前不能进入训练与正式比较：

- SpotSound 官方仓库 `d60c0e214ab685d0aa916db655fc59191900fb92` 只包含 Audio Flamingo 3 的模型、processor、训练和推理实现；
- 官方 Hugging Face `Loie/SpotSound` 只发布一个约 80.8 MB 的 LoRA adapter，其 `base_model_name_or_path` 为 `audio-flamingo-3-hf`；
- 官方未发布 SpotSound-Q checkpoint、Qwen2-Audio 的 timestamp-interleaved processor 或完整 77.6k 训练数据；
- 当前服务器也不存在可核验来源的 SpotSound-Q 私有资产。

因此，无法构造“同一官方 SpotSound-Q checkpoint 上的 matched SFT 与 RelTwin”这一公平实验。直接从公开 Qwen2-Audio 加载现有小规模 RelTwin 数据训练，只能称为 `Qwen2-Audio + RelTwin`，不能称为 `SpotSound-Q + RelTwin`，且无法复现论文中 49.9/85.4/72.4 的官方起点。本计划将 Q 路线标记为 `blocked-artifact-missing`，不为追求结果而猜测官方实现；执行顺序转入第 7 节的 TimeAudio 跨底座扩展。

## 2. 固定 Benchmark 与论证职责

| Benchmark | 固定规模 | 论证职责 | 执行方式 |
| --- | ---: | --- | --- |
| SpotSound-Bench | 400 queries | 核心短目标时间定位能力 | 全量评测 |
| Clotho-Moment | 6,649 queries | 大规模跨数据分布迁移 | 固定前 1,000 条看趋势，通过门禁后全量 |
| UnAV-100 subset | 100 queries / 77 audios | 外部真实音频迁移 | 公开 manifest 全量评测 |

UnAV 当前持有的是 AMR 作者公开的完整 100-query 子集，100/100 均已核验。SpotSound 论文表中的 UnAV 规模与该公开子集不一致，因此最终只在同一公开 manifest 上比较本轮所有模型；论文表格中的原始数值只能作为参考，不能用于严格 SOTA 声明。

## 3. Q 主实验的模型与公平对照

| 模型 | 初始化 | 新增训练 | 论证职责 |
| --- | --- | --- | --- |
| SpotSound-Q official | 官方 SpotSound-Q checkpoint | 无 | 固定官方底座能力与推理口径 |
| SpotSound-Q + SFT | 同一官方 checkpoint | 普通继续微调 | 排除数据和训练预算本身带来的增益 |
| SpotSound-Q + RelTwin | 同一官方 checkpoint | SFT + RelTwin 关系目标 | 检验 RelTwin 的独立贡献 |

Q + SFT 与 Q + RelTwin 必须共享数据、样本顺序、训练步数、LoRA 参数预算、学习率、优化器、随机种子、rehearsal 比例、prompt、解码参数和评测器。唯一实验变量是 RelTwin 关系目标。

## 4. 固定训练配置

除非代码审计证明该配置与 SpotSound-Q 结构不兼容，否则沿用已经验证的 RelTwin 训练配方：

```yaml
relation_source: ESC-50
relation_classes: 40
source_recordings: 256
inverse_pairs: 512
relation_queries: 1024
rehearsal_samples: 512
updates: 256
peft: LoRA
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
learning_rate: 5.0e-6
optimizer: AdamW
weight_decay: 0.0
gradient_clip_norm: 1.0
relation_temperature: 1.0
sft_weight: 1.0
candidate_ce_weight: 1.0
exchange_js_weight: 1.0
rehearsal_weight: 0.5
seeds: [0, 1, 2]
checkpoint_selection: fixed_final_step
```

不得用测试集选择 seed、checkpoint 或超参数。正式结果报告三个预注册 seed 的均值、标准差和逐 seed 原始值；单 seed 只用于实现验证和趋势门禁。

## 5. 分阶段执行、审计与停止条件

### Q-P0：资产与协议审计

开始训练前必须完成：

1. 核验官方 SpotSound-Q 代码、checkpoint 来源、大小和 SHA-256；
2. 核验模型架构、processor、音频采样率、prompt 模板、时间单位与解析规则；
3. 核验三个 benchmark 的 manifest、音频路径、duration clipping 和评测器；
4. 检查 checkpoint state-dict 覆盖率，不允许静默漏载关键模块；
5. 固定仓库 commit、依赖版本和所有正式运行命令。

通过标准：资产完整，官方推理路径可执行，协议没有未知的实质性缺口。否则停止计算并记录阻塞点。

### Q-P1：实现与代码审计

实现 SpotSound-Q 的最小适配层，并在提交正式任务前自行审计：

1. 静态检查张量 shape、mask、时间坐标、候选序列和 loss 组合；
2. RelTwin 关闭时，输出和普通 SFT 路径保持一致；
3. RelTwin 开启时，新增参数确实参与前向并获得有限梯度；
4. 不修改既有 SpotSound-A 结果路径；
5. 普通 SFT 与 RelTwin 的参数量、数据量和训练预算匹配。

通过标准：单元测试、语法检查和差异审计全部通过；不存在 NaN/Inf、参数漏训或测试集泄漏。

### Q-P2：关键路径 smoke

按以下顺序运行：

1. 单样本官方 Q 推理；
2. 单 batch 的 SFT 前向与反向；
3. 单 batch 的 RelTwin 前向与反向；
4. 4--8 个 update 的双组训练；
5. 固定少量样本评测，检查输出有效率与时间解析。

通过标准：训练稳定、checkpoint 可保存和重载、评测输出一一对齐，解析失败不被静默丢弃。

### Q-P3：官方 Q 基线同协议重跑

1. SpotSound-Bench：400 条全量；
2. UnAV-100 subset：100 条全量；
3. Clotho-Moment：固定前 1,000 条。

记录 mIoU、R1@0.3、R1@0.5、有效输出率、解析失败率、运行时间和原始预测。若官方 Q 在统一协议下大量解析失败，先修复协议适配，不进入训练。

### Q-P4：seed 0 匹配训练与趋势门禁

从完全相同的 official checkpoint 分别训练：

```text
SpotSound-Q + SFT, seed 0
SpotSound-Q + RelTwin, seed 0
```

训练结束后执行：SpotSound 400、UnAV 100、Clotho 固定前 1,000 条。

继续多 seed 的条件为满足以下任一项，且不存在超过 1.0 mIoU 的明显跨集退化：

- RelTwin 相对 matched SFT 在至少两个 benchmark 上提升 mIoU；
- RelTwin 在一个 benchmark 上提升至少 0.3 mIoU，并在另外两个 benchmark 上不下降；
- R1@0.5 提升至少 0.5，同时 mIoU 不下降。

若没有正信号，先审计数据流、梯度、prompt 和 checkpoint 加载；实现无误后停止 Q 调参，不在测试集上反复搜索。

### Q-P5：三 seed 与全量评测

通过 Q-P4 后补跑 seeds 1/2，并对三个 seed 的固定 final checkpoint 进行：

1. SpotSound-Bench 400 全量；
2. UnAV-100 100 全量；
3. Clotho-Moment 6,649 全量。

最终保存：逐 seed 指标、均值与标准差、逐查询 paired delta、以音频为重采样单位的 bootstrap 95% CI、有效输出率、解析失败率、训练/推理时间和 checkpoint 哈希。

## 6. Q 主实验结果表

| Model | SpotSound mIoU | Clotho mIoU | UnAV-100 mIoU | 有效输出率 |
| --- | ---: | ---: | ---: | ---: |
| SpotSound-Q official | TBD | TBD | TBD | TBD |
| SpotSound-Q + SFT | TBD | TBD | TBD | TBD |
| SpotSound-Q + RelTwin | TBD | TBD | TBD | TBD |

| Benchmark | RelTwin − SFT mIoU | 95% CI | R1@0.3 delta | R1@0.5 delta | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| SpotSound-Bench | TBD | TBD | TBD | TBD | TBD |
| Clotho-Moment | TBD | TBD | TBD | TBD | TBD |
| UnAV-100 subset | TBD | TBD | TBD | TBD | TBD |

所有 `TBD` 只能由实际运行结果填写，不用推测值、最佳 seed 或不同协议的论文数字替代。

## 7. Q 完成后的 TimeAudio 扩展

Q-P5 完成后，接入公开 TimeAudio checkpoint，执行与 Q 相同的审计链：

```text
资产与协议审计
→ 适配层实现与代码审计
→ 单样本/单 batch smoke
→ TimeAudio official 同协议重跑
→ matched SFT 与 RelTwin seed 0
→ 趋势门禁
→ seeds 1/2 与全量评测
```

TimeAudio official、TimeAudio + SFT、TimeAudio + RelTwin 仍使用相同数据、LoRA 参数预算、训练步数和评测器。前三个 benchmark 保持不变。若时间和数据资产允许，再增加 TimeAudio 原生采用的 AudioGrounding-v2 official 997-query test，作为跨论文官方环境的补充证据；该结果不得与 SpotSound 论文中另一种 AudioGrounding 规模混合。

TimeAudio 扩展进入论文主线的标准：RelTwin 相对 matched SFT 在至少两个 benchmark 上取得正向结果，且其余 benchmark 没有明显退化。否则只作为补充实验，不扩大“跨底座通用性”主张。

## 8. 运行管理与阶段审计

每个阶段固定执行：

```text
实现代码
→ 静态/差异审计
→ 最小 smoke
→ 审计日志与产物
→ 正式运行
→ 指标、协议和异常复核
→ 决定进入下一阶段或停止
```

长时间训练和评测统一放入可恢复的 `tmux` 会话，日志、PID、配置、checkpoint 和原始预测保存在项目目录。设置低频监控，仅在阶段完成、任务失败、GPU 异常或需要用户决策时通知；不持续占用交互会话轮询。

## 9. 最终论文判断

- A 与 Q 都成立：支持“RelTwin 跨 SpotSound 架构稳定有效”；
- A、Q、TimeAudio 都成立：支持“RelTwin 可迁移到不同自回归音频语言模型”；
- Q 仅单一数据集成立：保留为补充证据，主张收缩到该场景；
- 公平 matched SFT 下无稳定收益：停止扩张主张，保留已经被现有结果支持的 SpotSound-A 结论。
