# 新主线稿：作者审查与证据边界

本文件不参与编译。此次只改论文表达，不新增实验；旧版正文与结果文件均保留。

## 本次已经核实

- RBEE 三种子平均 mIoU 59.166010%，比同评测 official 高 0.848785 点。
- 同 seed 0：RBEE 59.183894%，SFT 58.313494%；记录中的训练步数、学习率、配对数量、回放数量和可训练参数数量匹配。此处 SFT 也使用 RelTwin 配对数据，不能称为完全不含关系样本。
- SetPO 相对 RBEE 在公开集的平均增益仅 0.026649 点；开发集 +3.357689 点不能挪作公开 benchmark 的增益。
- RelTwin 开发数据为 80 条合成音频、320 条查询、160 个模板条件下的关系对。按类别与本次适配训练隔离，不保证与骨干预训练数据隔离。
- 关系诊断旧 bootstrap 脚本对各个种子分别进行 query-row 或 relation-pair 重采样，然后取均值。不是 audio-group bootstrap，也未保持跨种子的相同抽样索引；不能扩大其统计解释。
- 主表完整系统 SpotSound 59.385511%，Clotho 86.855592%，UnAV 73.464610%。Clotho 和 UnAV 使用 official 初始预测，不是 RBEE/SetPO 迁移测试。

## 投稿前仍需解决，不能写成已完成

- 对 RBEE／RelTwin 的核心机制开展针对性近邻文献核查；之前针对动态半径的碰撞检查不等价于这项新主线的核查。当前仅核对引用文献，不宣称“首次”或“没有同类工作”。
- 区分候选判别与交换正则的独立作用。已有单种子 SFT 对照支持组合训练目标，但不足以证明 JS 交换项独立有效。
- 检查关系能力向真实音频和自然查询的迁移；合成双事件模板与通用时序推理不是同一个结论。主表收益与关系诊断收益之间尚无完整因果归因。
- 若需要显著性声明，重审共享音频、模板、种子相关性及开发选择效应；不得将旧区间重新命名成源音频分组置信区间。
- UnAV/AudioGrounding 专用清单与端点约定仍未完全对齐；AudioGrounding/AEGBench 参与过半径选择，不能恢复独立测试身份。DESED/TUT 有评测组装限制。
- 核对英文作者姓名、顺序、单位和邮箱，指定通讯作者，准备 ORCID。未知学院、ORCID 或通讯作者不可编造。
- 作者逐项审阅中英文公式、结果和表述；Acknowledgments 已披露文稿辅助，未声称作者完成了尚未发生的审核。
- 修改后重新检查 4 页技术内容、最多第 5 页参考文献、最小字号、字体嵌入及投稿系统校验。

## 方法映射

`build_esc50_reltwin.py` → 同音频两个相反顺序、两种查询模板、40/10 类别划分、普通查询回放。

`train_reltwin_micro.py::outer_objective` → 平均序列分数、两候选交叉熵、JS 交换一致性。

`setpo_candidates.py`、`interval_metrics.py`、`setpo_objective.py` → 六候选、置换、等权 set-IoU 与软 F1、列表交叉熵。

`analyze_reltwin_probe.py` → PairAcc 及包含 tie 的 Swap error 定义。

`summarize_reltwin_multiseed.py` → 历史 bootstrap 的实际抽样单位。

新增 `validate_reltwin_bundle.py` 只读取冻结结果并检查文稿，不执行训练、推理或新的统计评测。
