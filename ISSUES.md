# RelTwin 论文长期维护清单

当前论文：[paper.pdf](paper.pdf)

目标会议：ICASSP 2027

当前稿：v13（P0-01 标题与主线统一稿）

清单审计日期：2026-09-14

论文提交截止：2026-09-16（以 [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) 为准）

> 当前策略：不增加新 benchmark，不重新训练，不改动已有实验数值。先把现有证据组织成一篇问题明确、方法必要、结果闭环的论文。

---

## 0. 这份清单如何维护

`ISSUES.md` 是本分支唯一的问题台账。论文源文件、`paper.pdf` 与本清单同步维护，不再创建平行审计文档。

### 0.1 优先级

| 级别 | 含义 | 处理顺序 |
|---|---|---|
| **P0** | 会直接影响创新性判断、技术可信度或投稿合规；不处理就不应提交 | 立即处理 |
| **P1** | 不一定构成硬伤，但明显影响论文成熟度、可读性和说服力 | P0 后处理 |
| **P2** | 局部措辞、版面和终稿一致性问题 | 全文定稿后处理 |
| **P3** | 需要新实验的未来增强项；当前截稿版本不执行 | 延后，不阻塞本稿 |

### 0.2 状态

只使用以下状态：

- `待改`：当前版本仍存在，作者或编辑者可以直接处理；
- `待作者`：缺少真实作者信息或声明，不能代填；
- `待终检`：正文定稿后才能完成；
- `延后`：当前明确不执行，不应反复写进正文自我否定；
- `已完成`：已有对应文件或 PDF 证据，不再列入活动待办；
- `接受限制`：属于真实证据边界，集中声明一次即可。

### 0.3 更新规则

1. 每个问题只保留一个稳定 ID，不再按章节另写一份重复清单。
2. 只有实际修改已出现在 LaTeX 或 PDF 中，状态才能改为 `已完成`。
3. 修改问题状态时同时填写验证依据；不要把“计划修改”写成“已完成”。
4. 新问题先判断是否与已有 ID 同义；能并入就不新增。
5. 历史版本由 Git 保存，正文只保留当前有效待办和必要追溯映射。
6. 论文文字、表格或图发生变化时，同步提交 LaTeX 源码、重新编译的 `paper.pdf` 和本清单。
7. 仅修改本清单时，明确记录 `paper.pdf` 未变。

### 0.4 当前总览

| 优先级 | 活动问题 | 当前判断 |
|---|---:|---|
| P0 | 8 | 其中 6 项为论文内容，2 项涉及作者信息与提交合规；P0-01 已完成 |
| P1 | 8 | 主要是结果解释、图表叙事、Scope、结论和版面 |
| P2 | 3 | 最后一轮语言与 PDF QA |
| P3 | 4 | 全部需要新实验，当前延后 |

---

## 1. 全文冻结的核心故事

这不是额外待办，而是后续所有改稿都必须服从的统一口径。

### 1.1 一句话科学问题

> When inverse relations co-occur in the same recording, temporal grounding requires more than detecting the constituent events: each query must be bound to its own locally correct window.

即：当两个逆序关系同时出现在一段录音中，困难不再是判断声音是否存在，而是把每个查询绑定到其描述的局部窗口。

### 1.2 RelTwin 的关键区别

RelTwin 的负答案不是其他录音中的随机窗口，也不是音频中不存在的事件。它是：

> 同一录音中真实存在、声学上有效、甚至是另一条查询的正确答案，但对当前查询而言关系错误的时间窗口。

全文优先统一为以下一种表述：

> **locally valid timestamp contrasts**

需要更完整解释时使用：

> **globally valid but query-conditionally wrong timestamp answers**

不要在 `hard answer negative`、`locally confusable negative`、`competing window`、`opposite answer` 等多个名称之间频繁切换。

### 1.3 三层贡献

1. **问题层**：识别 `query–window binding error`——模型知道事件存在，却把不同关系查询绑定到同一时间窗口。
2. **方法层**：构造同录音内的逆关系最小对照，并比较完整时间戳答案的查询条件似然。
3. **证据层**：paired diagnostic 证明 same-answer collapse 明显减少；timing intervention 表明固定静音布局不能完全解释增益；SpotSound-Bench 获得新的公开主表最佳点估计。

### 1.4 当前允许的最强结论

- 可以说 RelTwin 在引用的 SpotSound 主表口径上取得最高列出的点估计：mIoU 59.16、R1@.3 77.17、R1@.5 62.08。
- 可以说相对论文主表最佳 57.9，mIoU 提升 1.26 点。
- 可以说 matched seed-0 中 RelTwin 相对 SFT 提升 1.02 mIoU，音频组 bootstrap CI 为 `[0.23, 1.81]`。
- 可以说相对本环境 official re-evaluation 的三种子均值差为 +0.75，但其 CI `[-0.24, 1.74]` 跨零。
- 不得把“主表点估计 SOTA”扩写成“对 official 复评统计显著优于”。

---

## 2. P0：提交前必须解决

P0-01 已完成，具体修改与合理性审查见第 7 节。其他条目不因本轮叙事调整而自动关闭。

### P0-02 · Related Work 没有沿“RelTwin 与最近邻方法的结构差异”组织

- **状态**：待改
- **位置**：`sections/introduction.tex` 第 2–3 个相关工作段
- **当前问题**：Pengi、Qwen2-Audio、Audio Flamingo 3、TimeAudio、SpotSound、CLAP、T-CLAP、CompA、CoSTALA 和 SHINE 连续出现，但缺少分类轴；读者知道工作很多，却不知道 RelTwin 填了什么空白。
- **为什么重要**：当前创新最容易受到的质疑不是模型性能，而是“hard negative、顺序反转和组合对比早已存在”。如果不主动区分监督对象与对比位置，novelty 会被压扁。
- **处理方向**：按三类组织：
  1. 时间表示和音频时序定位；
  2. 组合推理与顺序反转；
  3. 反事实或 hard-negative 对齐。
- **必须明确的结构差异**：
  - T-CLAP/CompA 等主要在表示或音频—文本匹配层面对比组合关系；
  - SHINE/AHA 类方法使用负查询或反事实回答抑制不一致；
  - RelTwin 在同一录音内部，让生成模型在两个都具有真实声学依据的完整时间戳答案之间进行查询条件竞争。
- **候选补充工作**：AHA、Auto-AEG、TAG-Bench、LAT-Audio 只作为定位候选；正式加入前应逐篇核对任务、监督形式和可支持的差异，不能只为显得“新”而罗列。
- **验收标准**：相关工作段最后能自然得到以下缺口，而不是突然宣布方法：现有方法尚未直接训练生成式 grounding 模型在同一录音的两个真实局部答案之间依据关系查询作选择。
- **来源映射**：新增包装审计第 5 节；原审阅 #13–#20。

### P0-03 · 方法动机没有说明为什么普通 SFT 不足

- **状态**：待改
- **位置**：`sections/introduction.tex` 方法概述；`sections/method.tex` Section 2.2
- **当前问题**：正文直接从 `Candidate cross-entropy` 跳到“rewarding correspondence”，没有先定义 SFT 缺少的相对偏好约束。
- **为什么重要**：没有这一层，RelTwin 看起来只是给已有生成训练附加一个普通二分类 loss，而不是针对 query–window binding failure 的训练原则。
- **处理方向**：明确三步逻辑：
  1. sequence supervision 分别提高每条正确时间戳序列的似然；
  2. 它不要求当前查询的正确窗口压过同录音内另一条真实窗口；
  3. RelTwin 把两个窗口组成局部候选集，并施加行内相对偏好。
- **应让读者直接看懂的约束**：
  - `s(q_+, W_+) > s(q_+, W_-)`
  - `s(q_-, W_-) > s(q_-, W_+)`
- **验收标准**：审稿人不依赖 Figure 1，也能解释 `L_cand` 比 `L_seq` 多提供了什么监督；正文同时说明训练时比较候选、推理时仍直接生成时间戳。
- **来源映射**：新增包装审计第 6、9 节；原审阅 #27–#32 中仍适用于当前方法的部分。

### P0-04 · 关键对象和损失定义仍有歧义

- **状态**：待改
- **位置**：`sections/method.tex` 全节，重点是 Section 2.1–2.3
- **当前问题**：
  - `Each spans the entire ordered sequence` 容易被误解为两个目标覆盖整段录音；
  - `Both queries are present at recording level` 把 query 与 query 描述的关系混为一谈；
  - `y(S)` 没有具体说明时间戳答案的序列化形式；
  - `L_seq` 和 `L_replay` 在总损失中出现但没有定义；
  - category-query rehearsal 的作用及其与 SFT 对照的关系没有讲透；
  - `Y=I_2`、Figure 1(b) 与 loss 公式没有被统一解释为一个 2×2 score matrix；
  - Swap error 的文字定义需要回看上一句才能判断。
- **为什么重要**：这些不是单纯英语问题，而是会让 reviewer 无法确认候选、正负标签、监督单位和对照公平性。
- **处理方向**：
  1. 明确每个 `W` 是对应局部有序事件对的起止范围，而不是整段录音；
  2. 明确两种关系在录音层面都成立，但各自正确区间不同；
  3. 给出 `y(S)` 的最小格式示例或明确其与 SpotSound 时间戳序列化一致；
  4. 一句话分别定义正确答案 sequence loss 与类别 rehearsal loss；
  5. 用 2×2 score matrix 统一候选集合、`Y=I_2` 和 row-wise CE；
  6. 直接用条件式定义 swap error，并注明 tie 计错。
- **验收标准**：只读方法节即可回答“录音如何构造、每个 query 的正负答案是什么、三个 loss 各做什么、SFT 与 RelTwin 唯一区别是什么、推理需要什么输入”。
- **来源映射**：C02、C03、W03；原审阅 #21–#35；新增包装审计第 8–10 节。

### P0-05 · 核心结果没有形成“失败模式 → 绑定恢复 → 公共迁移”的证据链

- **状态**：待改
- **位置**：`sections/bridge_results.tex`；`sections/results.tex`；`tables/training_results.tex`
- **当前问题**：结果段第一段同时堆叠 relation mIoU、JointPairAcc、Swap error、SpotSound mIoU、CI 和 92/239/69，读者很难判断哪个结果直接验证核心命题。
- **为什么重要**：本文最有区分度的证据不是单一 mIoU，而是模型从“两个逆查询返回同一答案”转向“按查询选择不同正确窗口”。数字顺序若不服务这个故事，实验容易被看成普通小幅刷榜。
- **处理方向**：按以下阶梯叙述：
  1. **失败模式**：SpotSound-A/SFT 是否产生 same-answer collapse；
  2. **关系选择**：SFT 的 identical-answer 37/160 降至 RelTwin 的 1/160，JointPairAcc 提升 31.25 点；
  3. **边界质量**：relation mIoU 提升 13.97 点，同时诚实保留案例中的边界误差；
  4. **公开迁移**：matched seed-0 SpotSound +1.02，再报告三种子主表点估计 59.16。
- **待核小项**：已有缓存中两条 IoU 下降至少 0.5 的样本应先核对错误类型；若不能形成明确发现，删除该计数而不是猜测归因。
- **验收标准**：结果第一段先回答“核心失败是否被修复”，表格和正文不重复所有数字；92/239/69 等次级统计只有在解释具体发现时才保留。
- **来源映射**：C04；原审阅 #51–#64；新增包装审计第 12、16 节。

### P0-06 · 论文仍大量使用内部审计和开发日志式语言

- **状态**：待改
- **位置**：`sections/abstract.tex`、`sections/experiments.tex`、`sections/bridge_results.tex`、`tables/training_results.tex`、Figure 1 caption
- **当前残留**：
  - `same-environment, equal-update seed-0 comparison`
  - `Current SFT, s0`
  - `SpotSound-A (re-eval.)`
  - `final checkpoints and including every seed`
  - `verify row alignment, audio identity, duration, and recomputed IoU`
  - `Code, inputs, weights, and environment are hashed before inference`
  - `the relation split and public scores have informed project development`
  - 图注中的内部查询编号与 `seed 0`
- **为什么重要**：这些表述强调“我们如何核账”，而不是“实验为何能回答科学问题”，使稿件像内部开发报告，也反复提醒 reviewer 证据有限。
- **处理方向**：
  1. 公平性条件集中到 Experimental Setup 一次；
  2. 哈希、逐行核验和 checkpoint 清单留在代码仓库或复现记录，不占论文正文；
  3. 表格行名改为方法身份，不使用 `Current`、`s0`、`re-eval.`；
  4. 开发集用途保留一次准确说明，不在摘要、表注、结果和 Scope 重复；
  5. 图注只描述输入、监督和观察到的现象。
- **验收标准**：全文搜索不再出现无必要的 `current`、`re-eval.`、`cached`、内部样本 ID、hash、row-alignment、final-checkpoint 等工程过程词；必要公平性边界仍可在设置中定位。
- **来源映射**：W04、W05、W06；原审阅 #36–#50、#55–#59、#66、#91–#96；新增包装审计第 13–15、22 节。

### P0-07 · 三种比较口径必须清楚分工，避免 SOTA 主张被反驳

- **状态**：待改
- **位置**：摘要；`tables/training_results.tex`；`tables/main_results.tex`；Results 4.1/4.3
- **当前正确口径**：

| 用途 | 结果 | 可以支持的结论 |
|---|---|---|
| matched seed-0：SFT → RelTwin | SpotSound 58.42 → 59.43，+1.02；CI `[0.23, 1.81]` | 候选监督在该匹配运行中的增益 |
| RelTwin 三种子均值 vs SpotSound 论文主表最佳 | 59.16 vs 57.9，+1.26 | 引用主表口径下新的最高点估计 |
| RelTwin 三种子均值 vs 本环境 official re-evaluation | 59.16 vs 58.42，+0.75；CI `[-0.24, 1.74]` | 正的点估计差，但该区间不排除零差异 |

- **当前问题**：Table 1 同时承担 matched ablation、backbone reference 和三种子摘要；Table 2 caption 又解释来源、delta、加粗规则和 seed，造成口径互相干扰。结果小节最后落在跨零 CI 上，也削弱了本节真正 headline。
- **处理方向**：
  1. Table 1 只服务 `backbone/paired SFT → RelTwin` 的机制与训练对照；
  2. Table 2 服务公开主表点估计比较；
  3. 正文一次说明 published 57.9 与本地 official 58.42 的身份不同；
  4. 公开结果段先写三种子 59.16，再在适当位置给 matched/official 边界，不用跨零 CI 收尾；
  5. 不把 +1.26 写成 loss 消融增益，也不写成统计显著优势。
- **验收标准**：任意数字都能回答“与谁比、为何可比、支持什么结论”；`SOTA` 仅修饰引用主表上最高列出的点估计。
- **来源映射**：C05；原审阅 #9、#40–#50、#68–#73、#84–#86；新增包装审计第 18 节。

### P0-08 · 通讯邮箱与全员 ORCID 尚未完成

- **状态**：待作者
- **位置**：`authors.tex`；投稿系统
- **当前问题**：Wei Xu 的通讯邮箱仍为 `[to be provided]`；尚未取得并核验三位作者的有效 ORCID。
- **为什么重要**：这是真正的提交阻断项。ICASSP 2027 要求所有作者有有效 ORCID，PDF 与投稿系统作者列表及顺序必须一致。
- **作者待办**：
  1. 提供 Wei Xu 的真实通讯邮箱；
  2. 提供 Zhicheng Tang、Yuehan Zhang、Wei Xu 的 ORCID；
  3. 核对作者顺序、单位、邮箱、题目、摘要和主题分类与投稿系统完全一致。
- **禁止事项**：不得猜测邮箱、ORCID 或单位信息。
- **验收标准**：PDF 不含占位符；全体 ORCID 可验证；投稿系统与 PDF 作者信息逐项一致。
- **官方依据**：[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
- **来源映射**：原 F02、F03。

### P0-09 · 披露、PDF 合规与正式提交仍待最终确认

- **状态**：待作者／待终检
- **位置**：`sections/declarations.tex`；最终 `paper.pdf`；投稿系统
- **当前情况**：Funding、conflict 和 `Compliance with Ethical Standards` 小节已经存在，但其事实只能由作者最终确认。当前记录为 Wei Xu 个人支付 GPU 费用、无机构或企业资助、无相关利益冲突；使用公开音频和程序化构造数据，无新增受试者实验。
- **当前风险**：
  - 个人支付 GPU 费用的表述在论文中较突兀，应确认会议是否需要如此具体，而不是为了“完整”保留异常信息；
  - 工具使用相关披露需由作者按实际情况完成；
  - 当前本地编译通过不等于投稿门户检查已经通过；
  - 投稿截止前还需确认文件名、字体嵌入、无页码、文件大小、作者信息和上传状态。
- **处理方向**：作者只确认真实事实；语言与位置按官方要求压缩。不得补造资助、伦理批准编号或豁免。
- **验收标准**：全体作者确认声明；最终 PDF 满足 5 页上限和第 5 页内容限制；门户检查通过并获得提交确认。
- **官方依据**：[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)、[ICASSP 2027 Policies](https://2027.ieeeicassp.org/about/sps-policies/)
- **来源映射**：原 F04、F06；新增包装审计第 21 节。

---

## 3. P1：显著影响成熟度和说服力

### P1-01 · Experimental Setup 应从审计记录改成科学协议说明

- **状态**：待改
- **位置**：`sections/experiments.tex`
- **当前问题**：`Data and split roles`、`primary control`、精确到 20,185,088 的参数量、逐项 QA、hash 和开发过程占据较大篇幅；同数据、同顺序、同更新等公平性信息多次重复。
- **处理方向**：
  1. 小标题改为数据构造与评估协议、训练设置和统计方法；
  2. 保留 LoRA、学习率、更新数、损失权重、解码设置等复现所需信息；
  3. 公平性用一句集中说明；另用一句诚实说明相同更新数不等于相同 FLOPs；
  4. 参数量可写为约 20.19M，避免工程日志感；
  5. 删除 hash、row alignment 和“包括每个 seed”等运行审计语言。
- **验收标准**：每句话要么定义数据/协议，要么给出必要配置或统计单位；内部 QA 信息不再挤占科学叙事。
- **来源映射**：W04；原审阅 #36–#50；新增包装审计第 13–15 节。

### P1-02 · Table 1 的方法层次和行名仍不够清楚

- **状态**：待改
- **位置**：`tables/training_results.tex`
- **当前问题**：`Current SFT, s0`、`SpotSound-A (re-eval.)`、`RelTwin, 3 seeds` 混用方法身份、运行身份与统计摘要；caption 同时解释环境、updates、backbone、seed、标准差和 Joint 缩写。
- **处理方向**：让表格视觉上呈现 `SpotSound-A → Paired SFT → RelTwin`，另将三种子汇总与 matched seed-0 对照清楚分组；将公平性细节移到设置；行名去掉 `Current` 和 `re-eval.`。
- **验收标准**：不读 caption 也能看出哪两行构成直接消融、哪一行只是 backbone reference、哪一行是三种子报告。
- **来源映射**：新增包装审计第 12 节；原审阅 #40–#47。

### P1-03 · Figure 1 应强调机制和能力恢复，而不是工程缓存与残余失败

- **状态**：待改文字；视觉结构已完成
- **位置**：`figures/reltwin_overview.tex`、`figures/overview_art.tex`、Figure 1 caption、Results 4.1
- **当前问题**：主图结构和配色已经改善，但图注仍直接暴露 `Queries 56--57 (seed 0)` 这类内部运行身份；panel (c) 的主要信息应是 SFT 对两个查询返回同一窗口，而 RelTwin 恢复 query-specific selection。源码注释中的 `cached` 属于复现信息，不是论文可见缺陷。
- **处理方向**：
  1. panel (a) 说明同录音中两种逆关系与不同正确窗口；
  2. panel (b) 与 2×2 score matrix、`Y=I_2` 和 row-wise candidate CE 对齐；
  3. panel (c) 先呈现“同答塌缩 → 正确关系绑定”，边界欠覆盖只中性注明；
  4. caption 删除缓存、内部样本身份和开发过程。
- **验收标准**：Figure 1 单独阅读即可理解问题、监督和一个真实恢复案例；不篡改时间戳、IoU 或残余边界误差。
- **来源映射**：W05；原审阅 #55–#57；新增包装审计第 11 节。

### P1-04 · Timing intervention 的结论对象仍需更集中

- **状态**：待改
- **位置**：`sections/results.tex` Section 4.2
- **当前证据**：80 段中 40 段波形未变，40 段改变时间布局；20.00 点的 JointPairAcc 优势来自真正改变的 40 段。原布局对应差距为 42.50 点。
- **当前问题**：逐条预测复现等实现核验抢占篇幅；段落容易被读成“已排除全部时间捷径”或“证明自然场景泛化”。
- **处理方向**：明确该实验只检验固定静音布局是否能完全解释增益。先写干预对象，再写 SFT 明显获益但 RelTwin 仍保持 20 点 binding 优势，最后给出有限且正面的机制结论。
- **验收标准**：读者能区分 40 段干预子集与 160 queries 全体统计；正文不声称排除了所有 shortcut。
- **来源映射**：C06；原审阅 #10、#18、#26、#65–#67、#89；新增包装审计第 17 节。

### P1-05 · 公开结果段和 Table 2 的 headline 不够突出

- **状态**：待改
- **位置**：`sections/results.tex` Section 4.3；`tables/main_results.tex`
- **当前问题**：最重要的 59.16 与 +1.26 后紧跟多种比较解释，段落最后落在跨零 CI；Table 2 caption 同时解释 published source、Table 3、三种子、delta、prior best 和 bold 规则。
- **处理方向**：本节第一句和表格视觉中心统一为“RelTwin 在引用主表三项指标上取得最高列出的点估计”；caption 只交代不可从表体读出的来源/统计；official re-evaluation 作为补充口径，不以其跨零 CI 结束段落。
- **验收标准**：headline 清楚，但没有把点估计扩大成显著胜出；表注长度明显缩短。
- **来源映射**：P0-07 的呈现子项；新增包装审计第 18 节。

### P1-06 · Scope 像拒稿理由列表，边界重复过多

- **状态**：待改
- **位置**：`sections/results.tex` Scope；Experimental Setup；表注
- **当前问题**：alternative hard negatives、matched multi-seed、natural relations、long-range relations、pretraining overlap 等限制集中列出，且部分边界在设置和结果中重复。
- **处理方向**：
  1. 开发集用途和 source separation 的真实范围放在 Experimental Setup 一次；
  2. Scope 只保留最关键的两个外部有效性边界；
  3. 不把未来可做实验逐项写成当前论文缺失清单；
  4. 不删除会改变结论含义的 CI、开发用途和比较对象。
- **验收标准**：必要边界各出现一次；正面发现之后不再固定追加“but does not prove...”式自我否定。
- **来源映射**：C07、W06；原审阅 #37–#39、#45、#50、#52、#73、#81–#87；新增包装审计第 19、22、23 节。

### P1-07 · Conclusion 没有落回最独特的科学洞见

- **状态**：待改
- **位置**：`sections/conclusion.tex`
- **当前问题**：`retaining an advantage` 未说明优势是关系选择/联合定位；`provides a practical training strategy` 过于通用，任何附加 loss 都可以这样总结。
- **处理方向**：按“事件存在不等于关系绑定 → RelTwin 学习局部有效答案之间的查询条件偏好 → paired diagnostic、timing intervention 和 public benchmark 支持这一点 → 推理流程不变”收束。
- **验收标准**：最后一句回到 query-conditioned binding，而不是泛化的“方法有效”；不重复堆三套比较数字。
- **来源映射**：W07；原审阅 #88–#90；新增包装审计第 20 节。

### P1-08 · 分页、浮动体位置和第五页内容需要统一整理

- **状态**：待改／待终检
- **位置**：PDF 第 2–5 页
- **当前问题**：Table 1 早于结果段首次讨论；第 3–4 页小节衔接仍不够自然；参考文献从第 4 页开始，不符合作者希望的“前四页技术内容、第五页集中放文献”布局。
- **事实边界**：这不是当前已确认的官方违规。官方允许前四页包含参考文献，第五页只能包含 references、funding acknowledgements 和 Compliance statement。
- **处理方向**：在完成 P0/P1 内容改写后统一重排，优先让表格靠近首次讨论，并让技术内容自然结束于第 4 页。不得靠缩小字号、过量负间距或恢复已删除支线填版。
- **验收标准**：阅读顺序自然；第五页内容符合官方限制；字号、页边距和可读性不受损。
- **官方依据**：[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
- **来源映射**：原 F01、F05。

---

## 4. P2：全文定稿后的语言与版面检查

### P2-01 · 全文术语与局部英语仍需统一润色

- **状态**：待改
- **位置**：全文
- **重点词组**：
  - `places inverse relations`
  - `locally confusable answer negative`
  - `answers the opposite query`
  - `controls a constant-position preference`
  - `fixes source material`
  - `retaining an advantage`
- **处理方向**：优先陈述具体事实和因果关系，减少名词堆叠、抽象动词和中文逻辑直译。统一 `query–window binding`、`locally valid timestamp contrasts`、`candidate supervision` 的用法。
- **验收标准**：同一概念全篇只有一个主名称；指标定义不依赖回看前文；不机械删除有实际含义的 `not`、`without`。
- **来源映射**：W02、W03、W06、W07；原审阅 #2–#16、#24–#35、#91–#96。

### P2-02 · 图表 caption、正文引用和视觉可读性终检

- **状态**：待终检
- **位置**：Figure 1、Table 1、Table 2 及相邻正文
- **检查项**：caption 不重复 Experimental Setup；所有缩写首次出现时可理解；图表在彩色和灰度下都能区分；字号不小于模板要求；正文先引用后出现或尽量就近；最高值加粗规则准确。
- **验收标准**：图表可独立阅读，但不承担完整方法段或复现日志；PDF 缩放到论文实际尺寸仍清楚。

### P2-03 · 最终 PDF、引用和占位符 QA

- **状态**：待终检
- **位置**：最终提交包
- **检查项**：5 页上限；第 5 页内容限制；US Letter/A4；字体嵌入；无页码；无 overfull box；无未解析引用；无 `[to be provided]`、TODO、内部样本 ID；PDF 与投稿系统题目、摘要、作者顺序一致；文件小于 5 MB。
- **验收标准**：本地编译和官方门户检查均通过，记录最终提交文件哈希和 paper ID。

---

## 5. P3：未来增强项，当前不阻塞投稿

这些项目都需要新增训练或评测。当前已明确来不及补 benchmark，因此统一标为 `延后`，不要在 P0/P1 中反复出现，也不要在正文中写成一长串自我否定。

### P3-01 · 第二个自然关系 benchmark

- **状态**：延后
- **价值**：检验 RelTwin 是否超出 SpotSound 与合成 relation development 的当前证据范围。
- **候选**：Clotho-Moment 或其他具有自然关系、多事件和可比评测协议的数据集。
- **触发条件**：只有下一版本明确扩展论文证据时再启动；不得把旧 Clotho 86.86 重新归于当前 RelTwin。

### P3-02 · matched SFT/RelTwin 多种子重复

- **状态**：延后
- **价值**：区分候选监督收益与单个 matched seed 的随机性。
- **触发条件**：要声称“多种子稳定优于 matched SFT”时必须补；当前只能使用已有 seed-0 匹配对照与 RelTwin 三种子摘要。

### P3-03 · 其他 hard-negative 构造的直接对照

- **状态**：延后
- **价值**：证明收益来自“同录音局部有效时间戳竞争”，而不只是任意额外负例。
- **触发条件**：要声称 RelTwin 独立优于 AHA、counterfactual query 或随机窗口负例时必须补；当前只陈述结构差异。

### P3-04 · 更自然的重叠、长距离关系或跨骨干验证

- **状态**：延后
- **价值**：扩大外部有效性，检验固定模板、短时关系和特定 backbone 之外的表现。
- **触发条件**：作为后续完整版本或期刊扩展，不挤占当前投稿。

---

## 6. 不得因“包装”而改变的事实

1. 当前论文核心方法是 **RelTwin**，对应历史上的 RelTwin-Cand `no_exchange` 配置，不是给旧完整系统换名。
2. RBEE、SetPO、旧路由和边界系统不属于当前核心方法，不恢复到主方法或主表。
3. 当前外部 benchmark 是 SpotSound-Bench；合成 relation development 是机制诊断，不得包装成第二个自然 benchmark。
4. SpotSound 主表比较使用论文主表最佳 57.9；本环境 official re-evaluation 58.42 是另一个比较口径，二者不能混写。
5. RelTwin 三种子 59.16 相对 57.9 是 +1.26 的点估计提升；相对 58.42 是 +0.75，且 CI 跨零。
6. matched seed-0 SFT 58.42 → RelTwin 59.43 的 +1.02 才是当前候选监督的直接匹配消融。
7. 相同更新数不代表相同 FLOPs 或 wall time；该事实集中说明一次即可。
8. relation split 与 public scores 参与过项目开发；不得写成从未接触的独立测试集。
9. adaptation/development 来源隔离不等于审计了基础模型全部预训练数据。
10. Figure 1 的真实时间戳、IoU、边界欠覆盖和残余失败不得为了视觉完美而修改。
11. SpotSound 原论文 Table 8 的 2 秒配置 87.2 不作为当前主方法口径；若未来恢复 Clotho-Moment 讨论，按作者决定只比较其默认/主方法 85.6，不能把 87.2 混成主表 SOTA。
12. 旧流程的 Clotho 86.86 不是当前 RelTwin 成绩，不得用于本稿跨 benchmark 声明。
13. 当前稿可以声称“引用主表上最高点估计”，不能声称所有公平重评口径下均统计显著领先。

---

## 7. 已完成事项

已完成项不再混入活动优先级；只有发生回归时才重新打开原 ID 或新增明确问题。

| ID | 状态 | 已完成内容 | 验证依据 |
|---|---|---|---|
| D01 | 已完成 | 补齐 LoRA rank 8、alpha 16、dropout 0.1、每步样本组成、梯度累积含义与解码配置 | `sections/experiments.tex`；v12 `paper.pdf` |
| D02 | 已完成 | 论文聚焦 RelTwin，删除 RBEE、SetPO、旧路由/边界系统与无关 Clotho/UnAV 支线 | 当前 LaTeX 与两张主表 |
| D03 | 已完成 | Figure 1 删除底部 inference pipeline 和整体 160 统计，panel (c) 改为 SFT/RelTwin 查询—窗口对照 | `figures/reltwin_overview.tex` 与当前 PDF |
| D04 | 已完成 | Figure 1 改为浅暖底与局部功能色，英文、公式、符号统一深色，减少大面积灰底 | 当前 Figure 1 与 PDF |
| D05 | 已完成 | 当前 PDF 可独立编译为 5 页 US Letter，无未解析引用和 overfull box | v12 构建记录；终稿后仍需重跑 P2-03 |
| P0-01 | 已完成 | 采用作者指定新标题，统一摘要、引言方法概述/贡献、关键词及结论主线 | v13 LaTeX、`paper.pdf` 与下方合理性审查记录 |

D01 的配置事实：

- LoRA rank 8、alpha 16、dropout 0.1；
- 每次更新使用一对逆关系查询和一条 category rehearsal query；
- SFT 与 RelTwin 使用相同数据、顺序和 256 次 optimizer updates；
- 16 kHz mono、bfloat16、greedy decoding、one beam、no sampling、最多 128 new tokens；
- 这些设置已写入当前稿，不需要重新训练。

### P0-01 · 标题、摘要与贡献主线统一（2026-09-14）

- **状态**：已完成。
- **采用标题**：`RelTwin: Contrasting Locally Valid Timestamp Answers for Query-Specific Audio Grounding`。
- **本轮范围**：只修改 `main.tex` 的标题与关键词、`sections/abstract.tex`、`sections/introduction.tex` 的方法概述和贡献段，以及为满足同一主线验收而同步的 `sections/conclusion.tex`。同时更新 PDF、README 与本清单。引言前三段（包括相关工作）原样保留；方法定义、训练配置、结果段、两张表、主图、参考文献、作者与声明均未修改。
- **已形成的主线**：同一录音内两种逆关系都成立 → 声音存在不足以决定当前查询对应哪个窗口 → 在完整时间戳答案间增加显式的查询条件比较 → 用关系绑定恢复、时间布局干预和 SpotSound 结果检验该做法。
- **摘要**：130 词，按失败模式、监督缺口、方法、机制证据与公共结果组织；关键词统一为 audio temporal grounding、timestamp answer contrasts、query–window binding、paired supervision。
- **贡献段**：分别陈述关系绑定失败的刻画、局部有效时间戳对比监督、以及 paired diagnosis/timing intervention/公共数据上的证据；不把普通交叉熵宣称为新损失范式。

**修改后的合理性审查：**

| 审查问题 | 结论与依据 |
|---|---|
| `locally valid` 是否会被误读为两个答案对当前查询都正确？ | 摘要和引言均说明：窗口有真实声学依据，但只有一个符合当前查询的顺序。与现有两个局部窗口及对角正标签一致。 |
| 是否把 SFT 说成不能学习关系？ | 没有。新文只指出 sequence supervision 没有显式比较两个完整候选答案，不否认 token-level 归一化中的竞争或 SFT 学到关系的能力。 |
| 是否混淆三个比较口径？ | 没有。31.25 与 1.02 明确对应 matched seed-0 paired SFT；20.00 对应实际改动时间布局的录音；59.16 和 +1.26 对应三种子均值与既有主表最佳。正文中 official 对照及跨零 CI 原样保留。 |
| 是否把合成开发集或时间干预包装为独立自然关系测试？ | 没有。摘要明确使用 synthetic relation-development data 和 timing-modified recordings；公共结果只点名 SpotSound-Bench。 |
| 是否夸大创新性、新增模块或推理能力？ | 没有声称首次提出对比学习、全面胜过其他 hard negatives，或引入新推理模块。直接生成时间戳与当前实现一致。 |
| 是否越界处理其他问题？ | 未重写相关工作、方法节、结果表或图。P0-02–P0-09 与全部 P1/P2/P3 状态保持不动；重叠措辞的后续审查以当前稿为准。 |
| 排版是否正常？ | 独立编译为 5 页，技术内容止于第 4 页；无 overfull box、未解析引用或交叉引用。新标题保持两行、原字号和模板不变。发现并修正关键词换行超栏；未处理 P1-08 的整体分页任务。 |

审查结论：**P0-01 可关闭；叙事与当前实现及已有证据一致。** 此结论不代替 P0-02 的相关工作核验，也不关闭 P0-03/P0-04 的方法动机和定义补全任务。没有训练、重新推理或新增实验。

来源映射：原审阅 #1–#12、#17–#20、W01、W02，以及新增包装审计第 2、3、6、7 节。

---

## 8. 原 96 条审阅意见到当前问题的追溯

原意见已在 Git 历史中完整保存。这里仅保留到当前稳定 ID 的映射，避免同一问题在正文重复维护。

| 原编号 | 当前去向 |
|---|---|
| #1–#12 | P0-01、P0-07、P2-01 |
| #13–#20 | P0-01、P0-02；已解决的旧句见 Git 历史 |
| #21–#35 | P0-03、P0-04、P2-01；已删除的旧扩展方法不再恢复 |
| #36–#50 | P0-06、P0-07、P1-01、P1-02 |
| #51–#64 | P0-05、P1-03；旧 exchange/setwise 分支已关闭 |
| #65–#67 | P1-04、P0-06 |
| #68–#73 | P0-07、P1-05、P1-06 |
| #74–#80 | 已完成：旧 routing/refinement、Clotho、UnAV 与无关精度记录已移出当前稿 |
| #81–#87 | P1-06、P3-01–P3-04；必要证据边界见第 6 节 |
| #88–#90 | P1-07 |
| #91–#96 | P0-06、P1-06、P2-01 |

---

## 9. 推荐修改顺序

按依赖顺序执行，避免先润色后重写造成返工：

1. **P0-02**：处理相关工作与 novelty gap；P0-01 的标题、摘要和贡献主线已完成，本轮未扩展到相关工作；
2. **P0-03 + P0-04**：补齐方法动机、对象、损失和指标定义；
3. **P0-05 + P0-07**：重排结果证据链和三种比较口径；
4. **P0-06 + P1-01 + P1-02**：清理内部审计语言并重整设置/Table 1；
5. **P1-03–P1-07**：统一图注、timing、public result、Scope 和 Conclusion；
6. **P1-08 + P2-01 + P2-02**：统一语言、图表和分页；
7. **P0-08 + P0-09 + P2-03**：作者信息、声明、最终 PDF 和门户提交检查。

P0-08 不依赖正文改写，作者应立即并行提供邮箱和 ORCID。

---

## 10. 完成标准

论文进入“可提交”状态前，至少满足以下条件：

1. Reviewer 能用一句话复述：RelTwin contrasts two locally valid timestamp answers whose correctness depends on the query。
2. Reviewer 不会把方法简化成“多加一个 CE”，因为正文明确解释 SFT 缺失的相对偏好约束。
3. Introduction 准确区分时间表示、组合/顺序对比、反事实 hard negatives 与 RelTwin 的同录音答案竞争。
4. 方法节独立定义录音、窗口、答案序列、三个损失、2×2 标签和三个 paired metrics。
5. 核心结果先讲 same-answer collapse 与 query-specific selection，再讲边界质量和公共 benchmark。
6. Table 1 一眼区分 backbone reference、matched SFT/RelTwin 和三种子摘要。
7. Table 2 的 59.16/+1.26 headline 清楚，同时不误称 official 对照统计显著。
8. 全文不再出现无必要的 `Current`、`re-eval.`、`cached`、hash、内部样本 ID 和 QA 清单。
9. 必要限制只集中出现一次，不把 P3 未来实验写成拒稿理由列表。
10. 结论落在“事件存在不等于关系绑定；局部真实答案仍需查询条件选择”。
11. 通讯邮箱、全员 ORCID、作者顺序和声明均由作者确认。
12. 最终 PDF 与投稿系统内容一致并通过官方检查。

---

## 11. 版本记录与可追溯信息

| 日期 | PDF 状态 | `ISSUES.md` 变化 |
|---|---|---|
| 2026-09-14 | v11，内容未变 | 建立 `paper` 文档入口并吸收 96 条外部审阅意见 |
| 2026-09-14 | v12 | 补齐关键训练/解码配置，C01 关闭 |
| 2026-09-14 | v12 | 重绘 Figure 1，删除 inference pipeline/整体统计并完成三轮配色调整 |
| 2026-09-14 | v12，PDF 未变 | 将置顶长篇审计与旧 P0/P1/P2 清单真正归并为单一长期台账；重定 P0/P1/P2/P3，建立稳定 ID、状态、验收标准和历史映射 |
| 2026-09-14 | v13 | 仅处理 P0-01：采用指定标题，统一摘要、引言概述/贡献、关键词与结论，并完成合理性和编译审查；其他问题状态不变，无新实验 |

- 当前 PDF SHA-256：`36241fa2b942d376e08e8b77e18fdc20c926cc8d6b62879e61d1ce31aa428bb5`
- v11 源起点：`3841a85d00ed4487be2cca7e1022b2279d079d79`
- 原交付审计提交：`a190b11`
- 原开发分支：`codex/reltwin-core-only-20260913`
- 当前论文入口：`main.tex`
- 当前固定 PDF：`paper.pdf`
- 官方规则最后核验：2026-09-14；提交前必须重新核对官方页面，不以本清单替代最新规则。
