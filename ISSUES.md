# RelTwin 论文问题清单

## 当前最高优先级：包装与表达专项修改（不补新 benchmark）

### 本轮范围

本文档只讨论当前 `paper.pdf` 在**不增加新 benchmark、不重新训练、不改变任何实验数值**的前提下，如何提升论文的逻辑、创新性呈现、故事性、方法解释和写作成熟度。

当前稿的主要问题不是“没有故事”，而是：

1. 最有价值的创新点没有被准确命名；
2. 方法的必要性没有讲透，容易被概括成“在合成数据上增加候选交叉熵”；
3. 大量实验过程、内部状态和防御性限定冲淡了核心贡献；
4. 已有结果没有被组织成一条逐层递进的证据链。

本轮修改目标不是把证据写得比实际更强，而是让读者优先看到本文真正解决的问题、方法为什么针对这个问题，以及现有实验已经支持了什么。

---

## 1. 全文应统一采用的核心故事

### 1.1 当前最强的科学命题

全文应该围绕下面一句话展开：

> When inverse relations co-occur in the same recording, temporal grounding requires more than detecting the constituent events: each query must be bound to its own locally correct window.

对应的中文逻辑是：

> 当两个逆序关系同时出现在一段录音中，困难不再是检测声音是否存在，而是把每个查询绑定到它所描述的局部窗口。

### 1.2 RelTwin 最值得强调的独特性

RelTwin 的负答案不是：

- 不存在于音频中的事件；
- 从其他录音随机抽取的窗口；
- 语言上伪造的错误描述；
- 只包含错误类别的简单负例。

RelTwin 的负答案是：

> 同一录音中真实存在、声学上完全有效、甚至是另一查询的正确答案，但对当前查询而言对应了错误关系的局部窗口。

这可以概括为：

> **globally valid but query-conditionally wrong timestamp answers**

或者更自然地写成：

> **locally valid timestamp contrasts**

全文应固定使用其中一个表达，不要在 `hard answer negatives`、`locally confusable negatives`、`competing windows`、`opposite answers` 等多个名称之间来回切换。

### 1.3 最合适的三层贡献

全文的贡献应按以下顺序展开：

1. **问题层**：发现并定义 query–window binding error，即模型知道相关事件存在，却把不同关系查询绑定到同一时间窗口。
2. **方法层**：构造同一录音内的逆关系最小对照，并在生成答案空间中比较完整时间戳序列的条件似然。
3. **证据层**：paired diagnostic 证明模型从 same-answer collapse 转向 query-specific selection；timing intervention 表明主要增益不能完全由固定静音布局解释；SpotSound-Bench 获得新的公开主表最佳点估计。

这三层组合起来，RelTwin 就不是“一个 loss”，而是“一个新失败模式、一种针对性训练原则和一套直接验证该能力的证据”。

---

## 2. 标题

### 2.1 具体位置

- 文件：`main.tex`
- 位置：第 8 行
- 当前标题：`RelTwin: Learning Query-Specific Windows from Co-Occurring Inverse Relations`

### 2.2 为什么当前标题不理想

`Learning Query-Specific Windows from Relations` 的动宾关系不自然，容易被理解为“从关系中生成时间窗口”。但方法并不生成候选窗口，而是在训练中学习查询与已有局部窗口之间的对应关系。

标题也没有体现本文最独特的部分：两个候选时间戳都来自同一录音、都具有真实声学依据，但只有一个与当前查询匹配。

### 2.3 建议修改方向

优先使用 `contrasting`、`binding`、`locally valid timestamps`，减少抽象的 `learning windows`。

首选方向：

> **RelTwin: Contrasting Locally Valid Timestamp Answers for Query-Specific Audio Grounding**

更稳妥的方向：

> **RelTwin: Learning Query–Window Binding from Co-Occurring Inverse Relations**

如果希望保留 inverse relations，同时突出方法机制：

> **RelTwin: Query–Window Binding through Paired Inverse-Relation Contrasts**

标题不建议出现 `hard negative`。这一表达过于宽泛，会把论文主动放进已经非常拥挤的 hard-negative 方法类别中。

---

## 3. 摘要

### 3.1 具体位置

- 文件：`sections/abstract.tex`
- 位置：第 2 行，整个摘要

### 3.2 当前摘要的主要问题

#### 问题一：开头描述了任务，但没有点出 SFT 的根本缺口

当前：

> Audio temporal grounding must distinguish occurrences that contain the same sounds in different orders.

这句话本身正确，但还没有告诉读者为什么现有训练会失败。真正的逻辑应该是：普通事件存在性或独立正答案监督可以识别相关声音，却不强制同一查询在两个真实窗口之间进行选择。

#### 问题二：`places inverse relations` 不自然

当前：

> RelTwin, which places inverse relations in one recording

关系本身不是被“放置”的物体。真正被构造的是包含两个逆序事件窗口的录音。

大致可改成：

> RelTwin constructs recordings in which both inverse relations occur in distinct local windows.

#### 问题三：`locally confusable answer negative` 是机器式名词堆叠

当前：

> providing a locally confusable answer negative

`answer negative` 不是自然搭配，也没有直接解释为什么这个负例特殊。建议直接描述事实：

> The competing timestamp is acoustically valid in the same recording but corresponds to the inverse query.

或者：

> This yields a locally valid timestamp contrast whose correctness depends on the query.

#### 问题四：摘要过度呈现实验审计条件

当前集中出现：

- `same-environment`
- `equal-update`
- `seed-0`
- `synthetic relation-development`
- 三个增益数字
- timing intervention 数字
- 三种子 SpotSound 数字

这些信息都是真实的，但同时放在摘要中会让核心结果看起来像一次受限的内部对照，而不是对新能力的验证。

摘要只需要保留两个最有决定性的结果：

1. JointPairAcc 提升 31.25 点，证明 query–window binding 被改善；
2. SpotSound-Bench 三种子均值为 59.16，高于公开主表 57.9。

`same initialization/data/order/updates` 等公平性信息应放在 Experimental Setup。timing intervention 可以用一句不带过多配置的概括。

#### 问题五：结尾太泛

当前：

> These results demonstrate the value of query-specific answer discrimination for generative audio grounding.

`demonstrate the value of` 是常见的泛化结尾，没有回到最独特的 scientific insight。

结尾应回到：即使多个候选窗口都包含正确声音，生成式 grounding 仍需要显式学习 query-conditioned timestamp preference。

### 3.3 建议的摘要结构

建议严格按五步组织：

1. **任务失败**：相同事件以相反顺序多次出现时，模型容易返回相同窗口。
2. **原因**：独立正答案监督不要求查询在多个真实窗口中进行相对选择。
3. **方法**：RelTwin 构造同录音逆关系对，并对完整时间戳答案进行候选似然竞争。
4. **证据**：JointPairAcc +31.25，timing 改变后优势保留，SpotSound 59.16。
5. **意义**：局部有效答案之间的查询条件竞争可以改善生成式音频 grounding。

---

## 4. Introduction 第一段：问题定义

### 4.1 具体位置

- 文件：`sections/introduction.tex`
- 位置：第 2 行

### 4.2 当前优点

`dog then rooster` 与 `rooster then dog` 是全文最清楚的例子。`query–window binding error` 也是一个值得保留的术语。这一段已经基本具备成熟论文的开头逻辑。

### 4.3 仍需加强的地方

当前最后一句：

> We target this distinction between event detection and query-specific localization.

它只说本文“关注”这个区别，没有把根因说出来。

建议在这一段结尾明确：如果训练只监督每个查询的正确答案，而不把同一录音中的另一个真实窗口作为竞争对象，模型仍可能主要依赖事件类别共现，而忽略关系词决定的局部对应。

大致方向：

> The failure arises because recognizing both events is sufficient for recording-level relevance, but not for deciding which local occurrence satisfies the queried relation.

这句话可以自然引出后面的 RelTwin，而不需要先提 loss 名称。

---

## 5. Introduction 相关工作与研究缺口

### 5.1 具体位置

- 文件：`sections/introduction.tex`
- 位置：第 4–6 行

### 5.2 为什么当前写法不理想

当前两段连续列出：Pengi、Qwen2-Audio、Audio Flamingo 3、TimeAudio、SpotSound、CLAP、T-CLAP、CompA、CoSTALA、SHINE。

主要问题不是引用数量，而是缺少分类轴。读者看完知道领域里有很多工作，却仍不清楚：

- 哪些方法主要解决时间表示；
- 哪些方法主要解决事件不存在时的幻觉；
- 哪些方法已经使用关系反转或 hard negatives；
- RelTwin 究竟填补了哪个尚未处理的空白。

### 5.3 建议的组织方式

#### 第一类：时间表示与时序定位

包括 TimeAudio、SpotSound、LAT-Audio 等。这类方法改善时间 token、长音频表示或 absent-event grounding，但没有显式解决同一录音中多个真实候选窗口之间的关系选择。

#### 第二类：组合推理和关系顺序

包括 T-CLAP、CompA。这类方法已经研究反转事件顺序、相同事件不同组合以及 composition-aware hard negatives。

必须准确承认这些相邻工作，否则 reviewer 很容易认为论文在忽略直接先例。

RelTwin 的差异应写成：

- CompA 在不同音频和描述之间做配对匹配；
- T-CLAP 在表示空间进行音频文本对比；
- RelTwin 在同一录音内部，对生成的完整时间戳答案进行查询条件竞争。

#### 第三类：反事实或 hard-negative 对齐

包括 SHINE 和 AHA。AHA 尤其重要，因为它已经把 temporal relation error 与 counterfactual hard negative 联系起来。

RelTwin 的差异不应写成“我们的负例更难”这种缺乏直接证据的判断，而应写成客观结构差异：

> AHA contrasts a grounded response with a counterfactual response that contradicts the audio. RelTwin contrasts two timestamp answers that are both grounded in the same recording, making correctness conditional on the query–window correspondence.

### 5.4 相关工作段最后应形成的缺口

建议以类似下面的逻辑收束：

> Existing approaches improve temporal representations, suppress absent-event predictions, or construct compositional negative queries. They do not directly train a generative grounding model to choose between two acoustically valid local answers when both queried relations hold in the same recording.

这是全文 novelty 最重要的一句话之一。

---

## 6. Introduction 方法概述

### 6.1 具体位置

- 文件：`sections/introduction.tex`
- 位置：第 8 行

### 6.2 当前问题

当前：

> Candidate cross-entropy supplements sequence supervision, rewarding the requested query–window correspondence.

这句话从方法名称直接跳到效果，没有解释为什么 sequence supervision 缺少这个能力。Reviewer 很容易把它读成“额外加了一个普通分类损失”。

### 6.3 建议修改方向

先说明 SFT 的缺口，再引出 candidate loss：

1. Sequence supervision independently raises the likelihood of each positive timestamp answer.
2. It does not require the positive answer to outrank another valid window in the same recording.
3. RelTwin converts the two inverse windows into a local contrast set and applies row-wise answer competition.

`Training compares candidates; inference retains timestamp generation.` 是好句子，可以保留，但应当放在方法概述末尾作为实际优势，而不是代替机制说明。

---

## 7. Introduction 贡献段

### 7.1 具体位置

- 文件：`sections/introduction.tex`
- 位置：第 10 行

### 7.2 当前问题

当前：

> Our contributions connect locally confusable answer construction, candidate-supervised adaptation, and a paired diagnostic...

`connect` 的语气太弱，像在拼接已有模块。它也没有告诉读者哪些是新发现、哪些是方法、哪些是评测。

后两句继续重复三个增益数字，却没有把每个实验对应到哪项贡献。

### 7.3 建议修改方向

建议明确写成三项贡献，但不需要夸张使用 `first` 或 `novel`：

1. We identify query–window binding as a failure mode distinct from event presence and boundary estimation.
2. We introduce paired, locally valid timestamp contrasts and optimize their query-conditioned likelihoods without changing inference.
3. We evaluate relation selection with a paired diagnostic and a timing intervention, and obtain a new best point estimate on SpotSound-Bench.

每项贡献都应对应后文一个明确部分：问题定义、方法、实验。

---

## 8. Method 2.1：录音与窗口构造

### 8.1 具体位置

- 文件：`sections/method.tex`
- 位置：第 3–5 行

### 8.2 `Each spans the entire ordered sequence` 有歧义

当前：

> Each spans the entire ordered sequence.

因为前一句同时出现 recording (x) 和 windows (W_+,W_-)，这里的 `entire ordered sequence` 可能被理解为完整录音，而不是某个局部的两事件窗口。

应该明确写出：

- (x) 包含两个互不重叠的局部窗口；
- (W_+) 从局部事件 A 的开始延伸到随后事件 B 的结束；
- (W_-) 从局部事件 B 的开始延伸到随后事件 A 的结束；
- 两者不是整条录音。

### 8.3 `Both queries are present at recording level` 主体错误

查询不会“存在于录音中”，存在的是查询所描述的两种关系。

大致可改为：

> Both inverse relations are true at the recording level, but their correct timestamp intervals are different.

这句话同时把本文最关键的条件讲清楚。

### 8.4 两个控制变量句子过于生硬

当前：

> Alternating the earlier relation controls a constant-position preference; excerpt reuse fixes source material within a pair.

问题：

- `controls a preference` 搭配不直观；
- `fixes source material` 容易被理解成修复素材；
- 一个分号承载了两个不同设计理由。

建议拆开说明：

> We balance which relation appears first to prevent a fixed-position shortcut. The two windows reuse the same source excerpts, so their distinction depends on order rather than event identity.

第二句比“控制声学材料”更直接地说明了为什么 reuse 有科学意义。

### 8.5 Rehearsal 的作用没有说清

当前只说 category queries target all occurrences for rehearsal，没有说明目的。

建议增加一句：rehearsal 保留基础的类别级定位能力，并在 SFT 与 RelTwin 中完全相同，因此不是两者差异来源。

不应声称 rehearsal 独立提升指标，因为当前没有对应消融。

---

## 9. Method 2.2：候选答案监督

### 9.1 具体位置

- 文件：`sections/method.tex`
- 位置：第 7–23 行

### 9.2 当前最大问题：只有公式，没有解释约束关系

Equation 1–3 数学上不复杂，但论文没有把它们与 SFT 的失败联系起来。

当前读者看到的是：

1. 定义平均 token likelihood；
2. 做一个 softmax；
3. 做 cross-entropy；
4. 加到原损失。

这会自然产生“方法只是普通候选 CE”的判断。

### 9.3 建议增加的关键解释

应明确给出 SFT 与 RelTwin 的差异：

普通 paired SFT 优化：

\[
\mathcal L_{\mathrm{seq}}
=-\tfrac12\left[s_\theta(x,q_+,W_+)+s_\theta(x,q_-,W_-)\right].
\]

它提高两个正确答案的绝对似然，但不要求：

\[
s_\theta(x,q_+,W_+) > s_\theta(x,q_+,W_-),
\]

也不要求：

\[
s_\theta(x,q_-,W_-) > s_\theta(x,q_-,W_+).
\]

RelTwin 的候选交叉熵正好施加这两个相对偏好。这个解释应在公式附近直接出现。

### 9.4 建议改成 (2\times2) score matrix 叙述

先定义：

\[
S_{ij}=s_\theta(x,q_i,W_j),\qquad i,j\in\{+,-\}.
\]

然后说明：对角项是 query-consistent timestamp answers，非对角项是同一录音中的 locally valid contrasts。候选损失对每一行做交叉熵，要求查询选择对应的对角窗口。

这样 Figure 1(b)、(Y=I_2) 和损失公式会形成统一解释，而不是三个分散元素。

### 9.5 (y(S)) 的输出形式需要具体化

当前只称为 `timestamp token sequence`，但没有说明实际文本形式。

建议给一个简短例子，例如：

> `from 3.250s to 6.800s`

同时说明 score 使用教师强制下的平均 token log-likelihood。平均化的目的是避免候选答案长度差异直接决定分数；如果两个候选始终等长，也可以只陈述这是与训练 sequence loss 一致的标准化得分，不要额外做无证据解释。

### 9.6 (\mathcal L_{seq}) 与 (\mathcal L_{replay}) 需要定义

当前 Equation 3 直接出现两个此前没有数学定义的项。建议至少用一句说明：

- (\mathcal L_{seq})：两个 query-consistent timestamp answers 的平均教师强制 NLL；
- (\mathcal L_{replay})：类别查询及其所有事件区间的教师强制 NLL。

### 9.7 删除或重写 no-grad/recompute 实现描述

当前 Experimental Setup 中写：

> RelTwin scores four query–answer combinations without gradients then recomputes weighted gradients...

这句话会让读者怀疑候选分数是否被 stop-gradient、目标是否与公式一致。

如果该实现只是显存优化，应大致改成：

> We compute the exact gradient of the four candidate scores by memory-efficient recomputation.

如果版面有限，这一实现细节可以完全留在代码中。方法正文只需要保证公式与实际训练一致。

---

## 10. Method 2.3：PairAcc、JointPairAcc 与 Swap Error

### 10.1 具体位置

- 文件：`sections/method.tex`
- 位置：第 25–28 行

### 10.2 当前优点

“两个查询都返回两个窗口的并集时，各自 IoU 可达到 0.5，但关系选择仍然错误”的例子非常好。它直接说明普通 IoU 为什么不足，应保留。

### 10.3 当前问题

`Swap error means that at least one query violates this strict preference` 需要读者回看前一句才能知道 strict preference 的数学条件。

建议直接定义：如果任一查询对自身目标的 IoU 不大于对逆关系目标的 IoU，则记为 swap error，包括平分。

此外，`Joint localization and relation selection` 可以进一步对齐全文术语，改成 `Paired query–window binding evaluation` 或类似标题。这样该指标更明显地服务于核心问题，而不是看起来像额外发明一个指标。

---

## 11. Figure 1

### 11.1 具体位置

- 文件：`figures/reltwin_overview.tex`
- 图内文字：`figures/overview_art.tex`
- PDF：第 2 页上方

### 11.2 Panel (a)

Panel (a) 应重点表达：

- 同一录音包含两种关系；
- 两个关系都真实；
- 正确答案是不同的局部窗口。

当前基本做到了，但 `Both relations hold in x; each query has a different target` 可以与正文统一成 `Both relations are true in x, but each query maps to a different local window.`

### 11.3 Panel (b)

当前图中有 (2\times2) 矩阵，但正文没有充分利用。修改正文后，panel (b) 应明确对应 score matrix (S_{ij})，对角格表示 query-consistent answers，非对角格表示 locally valid contrasts。

图中 `RelTwin candidate supervision` 可以改成更有方法辨识度的：

> Query-conditioned timestamp contrast

但最终名称必须与标题和正文保持一致。

### 11.4 Panel (c)

当前图内：

> Dev example 10007-v0; seed 0

caption 中：

> Queries 56–57 (seed 0)

这些都是内部实验索引，不提供科学信息，还会让图看起来像从调试日志直接导出。

建议统一改为：

> Representative inverse-relation pair

或者：

> Same-answer collapse and query-specific recovery

保留真实 target、prediction 和 IoU，不需要保留内部 ID 和 seed。

### 11.5 Caption

当前 caption 同时解释虚线、阴影、箭头、编号、seed、outline、filled bar 和 set-IoU，信息过载。

建议 caption 只完成三个任务：

1. 一句话说明整图结论；
2. 分别解释 a/b/c；
3. 保留必要视觉图例。

caption 不需要再说明 sample ID、开发缓存或工程来源。

---

## 12. Table 1：核心训练对照

### 12.1 具体位置

- 文件：`tables/training_results.tex`
- PDF：第 2 页右栏

### 12.2 当前行名的问题

当前：

- `Current SFT, s0`
- `RelTwin, s0`
- `SpotSound-A (re-eval.)`
- `RelTwin, 3 seeds`

`Current`、`re-eval.`、`s0` 都是实验管理语言。审稿人更关心方法关系，不关心它们在项目目录中的状态。

建议改成：

- `SpotSound-A backbone`
- `Paired SFT`
- `RelTwin`
- `RelTwin (3-seed mean)`

seed-0 条件可以放 caption 或表下注释一次，不必写进每一行方法名称。

### 12.3 当前 caption 的问题

当前 caption 使用：

> core training comparison

> separate references

这些词没有直接说明表格要证明什么。

caption 应突出能力递进：backbone、paired sequence supervision 和 query-conditioned candidate supervision 在相同 relation diagnostic 与 SpotSound 上的表现。

### 12.4 当前结果没有被充分解释

这张表最有价值的故事是：

1. SpotSound-A → Paired SFT：关系 mIoU 37.64→74.58，Joint 0.63→53.13；
2. Paired SFT → RelTwin：关系 mIoU 74.58→88.55，Joint 53.13→84.38；
3. SpotSound 上 Paired SFT 保持 58.42，而 RelTwin 提升到 59.43。

它说明：

- paired data 教会模型基本的关系定位；
- candidate competition 进一步解决同一答案塌缩；
- 这种关系训练没有牺牲公开 grounding，反而带来小幅提升。

正文应把这条阶梯明确写出来。

---

## 13. Experimental Setup：数据协议

### 13.1 具体位置

- 文件：`sections/experiments.tex`
- 位置：第 2–3 行

### 13.2 小标题像内部文档

当前：

> Data and split roles.

建议改成更常规的：

> Datasets and evaluation protocol.

### 13.3 `informed project development` 过于突出过程

当前：

> The relation split and public scores have informed project development.

这一事实关系到结果独立性，不能直接隐瞒，但 `project development` 很像内部交接记录。

大致可以更自然地集中说明：

> We use the constructed split for method development and report SpotSound-Bench as the public grounding evaluation.

如果 SpotSound 分数确实反复参与模型选择，则仍需保留准确边界，但不要在摘要、表注、结果和 Scope 中重复四次。

### 13.4 数据数字较密

当前一个段落连续出现 40 类、256 段、1,024 查询、512 对、512 rehearsal、10 类、80 段、320 查询、160 对、400 查询、387 音频。

这些数字有复现价值，但阅读负担大。可以把训练集和开发集压缩成紧凑括号，正文重点说明：

- 类别和 Freesound source disjoint；
- relation split 用于开发和绑定诊断；
- SpotSound 用于公开 grounding 评测。

---

## 14. Experimental Setup：训练对照

### 14.1 具体位置

- 文件：`sections/experiments.tex`
- 位置：第 5–8 行

### 14.2 小标题仍有审计语气

当前：

> Adaptation and the primary control.

`primary control` 像审稿回复。建议直接写：

> Training configuration.

或者：

> Paired SFT and RelTwin adaptation.

### 14.3 公平性信息重复过多

当前详细写了：

- same software environment；
- same code；
- same initial weights；
- same data order；
- equal updates；
- not equal FLOPs/wall time；
- 三种子统计与 matched seed-0 分离。

这些边界是必要的，但只需要一处紧凑说明。建议保留：

> The paired SFT and RelTwin runs share initialization, data order, and 256 optimizer updates.

然后用一句说明 RelTwin 增加候选打分，因此训练计算量更高。不要把相同事实重复放在摘要、caption、设置、结果和 Scope 中。

### 14.4 精确参数数量像日志

当前：

> 20,185,088 trainable adapter parameters

可以改为：

> 20.19M trainable parameters

精确数不是错误，但在四页论文中没有必要精确到个位。

### 14.5 `final checkpoints and including every seed` 是过程说明

当前：

> using final checkpoints and including every seed

这句话在内部审计中有意义，但对论文读者价值有限。直接写报告三种独立训练种子的均值和标准差即可。

---

## 15. Experimental Setup：不确定性与复现

### 15.1 具体位置

- 文件：`sections/experiments.tex`
- 位置：第 10–11 行

### 15.2 应删除的内部核验清单

当前：

> We verify row alignment, audio identity, duration, and recomputed IoU.

这是良好的内部实验实践，但不属于四页方法论文的主要科学内容。删除不会降低可复现性。

### 15.3 应删除的 hash 过程

当前：

> Code, inputs, weights, and environment are hashed before inference.

哈希不是本文研究对象，也不是 ICASSP 强制要求。它会强化“实验审计报告”的观感。论文更需要的是公开代码链接、数据构造说明和评测协议，而不是说明内部保存了 hash。

### 15.4 应保留的统计信息

以下内容有真实科学作用，应保留：

- public queries 按 audio group bootstrap；
- relation queries 按 source-connected group bootstrap；
- 查询保持成组抽样；
- 置信区间基于观测模型，而不是训练种子分布。

但最后一点无需用否定句反复强调。可以直接称其为 grouped sample uncertainty。

---

## 16. Results 4.1：核心机制结果

### 16.1 具体位置

- 文件：`sections/bridge_results.tex`
- 位置：第 1–4 行

### 16.2 当前第一段数字过载

当前同一段包含：

- relation mIoU 74.58→88.55；
- +13.97 和 CI；
- Joint 53.13→84.38；
- +31.25 和 CI；
- swap 40.00→7.50；
- SpotSound 58.42→59.43；
- +1.02 和 CI；
- 92/239/69；
- 两个下降至少 0.5 的样本。

数字已经替代了论证。读者难以判断最重要的发现是哪一个。

### 16.3 建议的结果叙事顺序

#### 第一层：先讲失败模式是否被修复

最有解释力的数字是：

> SFT returns identical answers for 37/160 inverse-query pairs, whereas RelTwin does so for only 1/160.

它直接对应 Introduction 定义的 query–window binding error。

#### 第二层：再讲是否绑定到正确窗口

使用 JointPairAcc 53.13→84.38，说明变化不是随机地产生不同答案，而是多数转向了正确关系窗口。

#### 第三层：最后讲边界质量与公开迁移

relation mIoU +13.97，SpotSound +1.02。这样从行为变化、关系正确性到普通 grounding 性能形成递进。

### 16.4 建议删除的统计

以下内容目前没有产生额外科学解释，可以删除：

- `92/239/69` win/tie/loss；
- `including two IoU decreases of at least 0.5`；
- `51 pairs change from incorrect to correct, one in the opposite direction, and 24 remain incorrect`。

如果不准备分析那两个大幅退化样本的错误类型，就不应该主动把它们放进主文。

### 16.5 因果表述需要更准确但不必防御

当前：

> Shared positive examples and update counts connect these gains to adding explicit competition...

`connect these gains to` 稍显绕。可以直接写：

> Because the two runs use the same paired examples and initialization, their difference isolates the effect of query-conditioned candidate supervision under the matched seed-0 setting.

该限定说一次即可。

---

## 17. Results 4.2：Timing intervention

### 17.1 具体位置

- 文件：`sections/results.tex`
- 位置：第 4–8 行

### 17.2 当前实验价值

这是当前稿最强的机制验证之一。它不是普通鲁棒性测试，而是针对一个具体替代解释：模型是否只是记住了固定静音和窗口位置。

### 17.3 当前写法的问题

第一段花费较多篇幅列出：1.5 秒 lead-in、0.25 秒 gap、3 秒 windows gap、2 秒 tail、40 条不变、40 条改变，以及不变样本逐条重现预测。

其中具体时间参数可以保留以保证复现，但 `reproduce every prediction` 是实现核验细节，不应成为结果段重心。

### 17.4 建议的故事

更好的逻辑是：

1. 统一重新合成时间布局，移除 relation 与固定位置之间的关联；
2. 冻结两个模型，不再训练；
3. SFT 因新布局明显改善，说明原布局确实提供了部分捷径；
4. RelTwin 仍领先 20 点，说明局部答案竞争带来的绑定能力不能完全由该捷径解释。

这一结果不需要写成“我们成功排除了所有 shortcut”。准确结论就是 `does not fully explain`，已经足够有力。

段落应以保留的 20 点关系选择优势结束，而不是以 `targeted, post-development test` 等审计身份结束。

---

## 18. Results 4.3：公开 SpotSound 结果

### 18.1 具体位置

- 文件：`sections/results.tex`
- 位置：第 10–11 行
- 对应表格：`tables/main_results.tex`

### 18.2 当前最重要的 headline

可以准确声称：

> RelTwin obtains 59.16 mIoU on SpotSound-Bench, exceeding the previously published main-table best of 57.9.

这应当是本节第一句和 Table 2 的主要信息。

### 18.3 不要让两种比较口径相互打架

当前同时存在：

- 与已发表主表 57.9 比：+1.26；
- 与本地相同评测器重新评估的 58.42 比：+0.75，CI 跨零；
- matched seed-0 SFT 58.42→59.43：+1.02，CI 为正。

三个数字都正确，但连续解释会让读者怀疑作者在选择有利参照。

建议给三者明确角色：

1. **Published comparison**：用于 SOTA point estimate；
2. **Matched SFT comparison**：用于候选监督的因果对照；
3. **Official re-evaluation**：作为评测器一致性参考，简短放一次。

不要把 +1.26 写成 candidate loss 的独立增益，也不要声称显著优于 official re-evaluation。

### 18.4 本节结尾不应是跨零 CI

当前段落最后一句是 official re-evaluation 的 CI 跨零。这会形成明显的负面终因效应。

可以先交代该统计边界，然后以更高 R1@.5 和 matched SFT 的一致方向收束。或者把 official reference 放进 table note，让正文结束在公开 benchmark 上保持 query-specific training 收益的结论。

### 18.5 Table 2 caption 需要缩短

当前 caption 同时解释 published baselines、Table 3、三种子、delta、prior best 和 bold 规则。

`Bold marks the highest listed point estimate` 没有必要。读者能理解粗体含义。

caption 只需说明：SpotSound-Bench、400 queries、published results from SpotSound、RelTwin three-seed mean。

---

## 19. Scope 小节

### 19.1 具体位置

- 文件：`sections/results.tex`
- 位置：第 13–14 行

### 19.2 当前问题

当前连续列出：

- repeated excerpts；
- fixed templates；
- alternative hard negatives 未评估；
- matched multi-seed replication 未完成；
- natural overlapping relations 未评估；
- long-range relations 未评估；
- backbone pretraining overlap 未审计。

这些限制大多真实，但集中放在主结果之后，等于替 reviewer 写了一份拒稿理由列表。

### 19.3 建议修改方向

只保留与当前结论直接相关的一条边界：

> The controlled construction isolates query–window binding under paired inverse relations; broader natural relation structures remain outside the present evaluation.

其他内部审计边界不需要全部进入四页正文。尤其 `does not audit backbone pretraining overlap` 会突然引入全文未讨论的数据污染问题，不建议主动扩展攻击面。

如果必须保留 SpotSound 参与开发这一事实，应放 Experimental Setup 一次，不要在 Scope 重复。

---

## 20. Conclusion

### 20.1 具体位置

- 文件：`sections/conclusion.tex`
- 位置：第 2 行

### 20.2 当前问题

当前结论相对简洁，但：

- `retaining an advantage` 没说明保留的是关系选择还是普通 IoU；
- `practical training strategy` 缺少训练效率数据支撑；
- 没有回到“两个候选都真实但只有一个与查询匹配”的核心洞见。

### 20.3 建议修改方向

结论应完成三件事：

1. 重申问题：事件存在不等于 query–window binding；
2. 重申机制：在生成答案空间对 locally valid timestamps 进行条件竞争；
3. 重申证据：减少 same-answer collapse、改善联合关系定位，并提高 SpotSound grounding，且不改变推理流程。

不需要再重复所有数值，也不要以未完成的自然场景泛化作为最后一句。

---

## 21. Acknowledgments 与伦理声明

### 21.1 具体位置

- 文件：`sections/declarations.tex`
- PDF：第 4 页

### 21.2 个人支付 GPU 费用表述异常

当前：

> GPU computing expenses were personally funded by Wei Xu.

这不是常见的论文致谢表达，也不会增强贡献或复现性，反而像项目财务备注。

作者应按真实情况和会议要求选择：

- 若确实没有外部资助，简洁写 `This work received no external funding.`；
- 若有学校、实验室或项目支持，则准确列出；
- 不要猜测或虚构资助来源。

### 21.3 利益冲突和伦理说明

`No relevant conflicts of interest` 与无人体实验说明可以保留，但应遵循官方要求，语言简洁即可。

这些声明可以与参考文献集中到第五页，释放第四页技术空间。

---

## 22. 全文需要清除的内部审计式表达

| 当前表达 | 为什么不合适 | 建议方向 |
|---|---|---|
| `Current SFT` | 内部版本状态 | `Paired SFT` |
| `SpotSound-A (re-eval.)` | 像实验记录 | `SpotSound-A backbone`，评测器信息放 caption |
| `Dev example 10007-v0; seed 0` | 内部样本索引 | `Representative inverse-relation pair` |
| `Queries 56–57 (seed 0)` | 调试标识 | 删除，只说明案例性质 |
| `Data and split roles` | 内部文档标题 | `Datasets and evaluation protocol` |
| `primary control` | 审稿回复语气 | `Paired SFT and RelTwin adaptation` |
| `using final checkpoints and including every seed` | 实验审计过程 | 直接写 three independent seeds |
| `row alignment, audio identity, duration` | QA 检查清单 | 从正文删除 |
| `hashed before inference` | 内部可追溯流程 | 删除，改为公开代码/协议说明 |
| `informed project development` | 项目管理语言 | 用科学的 development/evaluation protocol 表述一次 |
| `separate references` | 表格谱系说明 | 直接给每行科学名称 |
| `locally confusable answer negative` | 名词堆叠 | `locally valid timestamp contrast` |
| `fixes source material` | 搭配歧义 | `reuses the same source excerpts to control event identity` |
| `controls a constant-position preference` | 不自然 | `balances which relation appears first to prevent a position shortcut` |
| `retaining an advantage` | 指代模糊 | 明确 `retaining a 20-point joint binding advantage` |
| `demonstrate the value of` | 泛化模板句 | 直接写所观察到的机制结论 |

---

## 23. 哪些事实必须保留，不能为了包装而隐藏

包装不是删除所有不利信息。以下事实影响结论含义，必须保留，但每项只需在最相关的位置出现一次：

1. matched SFT–RelTwin 的核心因果对照是 seed 0；
2. RelTwin 三种子均值与 published SpotSound 主表比较，用于 point-estimate SOTA；
3. 与 58.42 official re-evaluation 的 +0.75 CI 跨零，不能写成统计显著优于；
4. relation split 是构造的 development diagnostic，不是第二个自然外部 benchmark；
5. RelTwin 训练增加了候选答案计算量，equal updates 不等于 equal FLOPs；
6. timing intervention 只排除了一个具体的位置/静音解释，不能写成排除所有 shortcut。

正确做法是集中、准确地表述这些边界，而不是在摘要、设置、表注、结果、Scope 和结论中反复自我辩护。

---

## 24. 推荐的全文逻辑顺序

### Abstract

问题 → SFT 根因 → locally valid timestamp contrast → candidate likelihood competition → binding 结果 → SpotSound 结果 → 核心意义。

### Introduction

1. 具体例子定义 query–window binding error；
2. 区分 temporal representation、absent-event negative 和 compositional hard negative；
3. 指出同录音两个全真局部答案仍未被解决；
4. 概述 RelTwin；
5. 列出 failure / method / evidence 三项贡献。

### Method

1. 构造包含两个逆关系窗口的同一录音；
2. 定义 (2\times2) query–answer score matrix；
3. 先说明 SFT 不提供相对约束，再定义 candidate CE；
4. 说明 inference unchanged；
5. 定义 paired binding metrics。

### Experiments

1. 数据和评测协议；
2. Paired SFT 与 RelTwin 的匹配设置；
3. grouped bootstrap，其他内部 QA 流程删除。

### Results

1. Backbone → Paired SFT → RelTwin 的能力阶梯；
2. same-answer collapse 37→1，Joint +31.25；
3. timing intervention 后保留 +20；
4. SpotSound 59.16 vs published 57.9；
5. 一句紧凑 scope。

### Conclusion

事件存在不等于关系绑定 → RelTwin 训练局部有效答案之间的查询条件偏好 → 现有诊断和公开结果支持这一结论。

---

## 25. 修改优先级

### P0：直接影响创新性和审稿判断

1. 改标题；
2. 重写摘要；
3. 重构 Introduction 相关工作和 novelty gap；
4. 重写贡献段；
5. 在 Method 中解释 SFT 缺失的相对偏好约束；
6. 补全窗口、(L_{seq})、(L_{replay}) 与 score matrix 定义；
7. 按 Backbone → Paired SFT → RelTwin 重写核心结果。

### P1：直接影响论文成熟度

1. 删除内部样本 ID、seed 标签和 `current/re-eval`；
2. 删除 row alignment、hash、final checkpoint 等过程语言；
3. 精简 Table 1、Table 2 和 Figure 1 caption；
4. 重写 timing intervention，使其表现为机制验证；
5. 压缩 Scope，避免拒稿理由列表；
6. 重写结论，使其回到核心 insight。

### P2：投稿前完成

1. 替换通讯作者邮箱占位符；
2. 核对资助、利益冲突和伦理声明；
3. 把声明和参考文献尽量集中到第五页；
4. 最后检查术语统一、引用、浮动体顺序和分页。

---

## 26. 修改完成后的验收标准

修改后的论文应达到以下效果：

1. Reviewer 能用一句话复述：RelTwin contrasts two locally valid timestamp answers whose correctness depends on the query。
2. Reviewer 不会把方法简单概括成“多加了一个 CE”，因为正文已经解释 SFT 缺失的相对约束。
3. Introduction 能主动、准确地区分 CompA、T-CLAP、AHA、SpotSound 与 RelTwin。
4. Table 1 一眼呈现 Backbone → Paired SFT → RelTwin 的能力递进。
5. 核心结果段先讲 same-answer collapse 和正确关系绑定，而不是先堆十几个数字。
6. 全文不再出现 `Current`、内部样本 ID、hash、项目状态和 QA 检查清单。
7. 必要实验边界只出现一次，不在每个正面结论后附加一段自我否定。
8. 结论最后落在“局部全真答案仍需查询条件绑定”这一科学洞见上。
9. 所有数值、置信区间和比较对象保持真实，不把 point-estimate SOTA 写成统计显著优势。

---

## 原有分级清单、合规事项与版本记录


当前论文：[paper.pdf](paper.pdf) · 目标会议：ICASSP 2027 · 更新日期：2026-09-14

**当前是 v12（C01 配置补充及 Figure 1 更新）审阅稿，不是已完成投稿检查的终稿。** 本分支以 `paper.pdf` 和 `ISSUES.md` 为固定阅读入口，同时保存可编辑的 LaTeX、图表、文献与模板。本次补齐已有实验的配置说明，并更新 Figure 1 的结构与配色；没有训练、重新推理或改动实验分数。旧版本由 Git 历史保存。编辑与编译方法见 [README.md](README.md)。

本清单按作者要求排序：**P0 格式与必要信息 → P1 论文内容 → P2 用语措辞**。P0 中也区分官方要求、作者排版要求和版面优化，不把三者混称为拒稿问题。`待改`、`待作者`、`待终检`是待办；`保留边界`是已知证据范围，不表示必须补实验；`已处理`不再重复修改。

合并来源：v11 的 PDF、LaTeX 与已有交付审计，以及作者提供的 `reltwin-icassp-writing-review.md`；随后按每轮改稿更新当前状态。该审阅文件含 **96 条意见，部分引用旧稿**。文末保留全部原编号的归类与处置，不能把 96 条全部当成当前缺陷。

## P0 · 格式与必要信息（优先处理）

### F01 · 正文与参考文献的分页不符合作者指定布局

- 状态：**待改；作者排版要求，不是已发现的官方超页违规。**
- 位置：PDF 第 4–5 页。
- 现状：v12 技术内容在第 4 页结束，但参考文献 [1]–[11] 已出现在第 4 页，第 5 页续列 [12]–[17]。尚未形成作者要求的“前四页正文、第五页集中放文献”。
- 修改：C01 的配置说明已补齐；后续结合必要结果解释，重新安排浮动体与参考文献起页。不能靠恢复旧系统支线、缩小字号或灌水填页。
- 验收：全部技术内容止于第 4 页；全部参考文献放第 5 页；保持模板、字号与可读性。
- 依据：官方允许最多四页技术内容，前四页也可以出现参考文献，并非规定必须写满四页正文。Paper Kit 另允许第五页放资助与伦理声明；本项目采用作者指定的更清晰布局。[官方 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)

### F02 · 通讯作者邮箱仍是占位符

- 状态：**待作者；必要联系信息。**
- 位置：第 1 页通讯作者脚注，`authors.tex`。
- 现状：Wei Xu 已标为通讯作者，邮箱仍为 `[to be provided]`。
- 修改：由作者提供并确认真实邮箱，替换占位。不能猜邮箱。
- 已确认信息：作者顺序为 Zhicheng Tang、Yuehan Zhang、Wei Xu；三人均为华中科技大学。前两位邮箱已经写入，不需重新占位。
- 验收：PDF 不含该占位，通讯作者与投稿系统资料一致。

### F03 · 全体作者 ORCID 与投稿元信息尚未完成核对

- 状态：**待作者／待投稿系统核验；不是已确认作者未注册。**
- 现状：尚未取得三位作者的有效 ORCID，也未核对投稿系统中的作者、单位、通讯资料、题目、摘要及主题分类。
- 修改：收集并核对三位作者 ORCID；保证 PDF 与系统作者名单、顺序一致，并同步最终题目与摘要。
- 验收：逐人确认有效 ORCID；系统填写完成且与最终稿一致。ORCID 必须提供给投稿系统，不等于一定要占用论文正文版面。
- 依据：2027 年所有作者均须提供有效 ORCID；缺失或 PDF/系统作者不一致会导致撤稿。[官方投稿要求](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)

### F04 · 作者负责的披露与声明终审尚未完成

- 状态：**待作者；合规确认。**
- 现状：资助、利益冲突和独立的 `Compliance with Ethical Standards` 小节均已存在，不是缺失项。当前记录为 Wei Xu 个人支付 GPU 费用、无机构或企业资助、无相关利益冲突；使用公开音频及程序化构造数据，无新增听评或受试者实验。
- 待办：作者核准声明与实际情况一致；不能补造伦理批准编号或未经确认的豁免。作者此前保留自行填写的工具使用相关披露，仍需在投稿前按实际情况完成；本次不代写声明，也不改动 PDF。
- 依据：资助来源、利益冲突或其不存在及伦理合规均需报告；生成式内容的披露与仅做语言编辑的适用情况不同，按实际用途判断。[会议政策](https://2027.ieeeicassp.org/about/sps-policies/)、[作者指南](https://2027.ieeeicassp.org/author-guidelines/)

### F05 · 表格位置与跨页断点有待整理

- 状态：**待改；版面质量，不是单独的硬性违规。**
- 位置：第 2 页表格、第 3–4 页小节衔接。
- 现状：Table 1 仍浮动到方法部分附近，早于结果段的首次引用。v12 中原先 4.3 标题带少量文字跨页的情况已因正常重排消失，但 4.2 末句仍跨第 3–4 页；整体浮动体与分页尚待 F01 一起整理。
- 修改：配合 F01 调整表格与首次讨论的距离，避免标题仅带一两行正文跨页。不要用大量负间距压缩版面。
- 验收：读者按方法、设置、结果的顺序能就近看到相应表格；分页自然。

### F06 · 最终 PDF 与投稿状态尚需终检

- 状态：**待终检；不是当前全部检查失败。**
- 现状：v11 的本地格式检查已通过下列项目，但未代作者完成投稿系统检查、全体作者终稿认可或正式提交。
- 修改：完成 F01–F05 后重新检查页数、字体、图表、引用与占位符，并由全体作者确认提交稿。
- 验收：新 PDF 通过本地与门户检查，登记最终文件哈希；取得投稿确认才标为已提交。
- 仓库文件名：按作者要求固定为 `paper.pdf`。实际投稿时再处理上传文件名要求，不改变仓库中的固定名称。

### 当前已经通过、不要误列为缺陷的项目

v11 已完成字体、模板与独立编译核验，交付审计记录 69 项测试通过。本次未重跑那 69 项历史测试。v12 重新编译为 5 页 US Letter，约 287 KB，技术内容止于第 4 页；摘要仍为 127 词，4 个关键词、17 篇文献。模板、字号、图表和结果数值未改；没有 overfull box 或未解析引用，已目视检查受重排影响的第 2–4 页。改稿后的本地检查不替代 F06 的最终门户核验。

## P1 · 论文内容（影响方法理解、归因与证据解释）

### C01 · 关键训练与解码配置（已补齐）

- 状态：**已处理，2026-09-14；无需训练。**
- 位置：Section 3，Adaptation and the primary control。
- 已补内容：LoRA rank **8**、alpha **16**、dropout **0.1**；每次优化器更新使用 **1 对逆关系查询（2 条关系查询）+ 1 条类别 rehearsal 查询**。梯度在这些损失分量之间累积，不跨查询对累积，不能把四次候选打分误报成四个独立样本的 batch。
- 已补解码：**单 beam 贪心解码、关闭采样、最多 128 个新 token/查询**，并说明 16 kHz 单声道输入和 bfloat16 推理。该配置共用于报告中的 SFT、RelTwin 与 official 复评。
- 核验依据：训练和推理脚本的 SHA-256 与两轮实验的冻结记录一致；已核查每步梯度清零、两条正例/四种候选组合、rehearsal 回传和优化器更新的执行顺序，以及实际启动参数。所有检查均为读取记录或 CPU 文件核验。
- LoRA 来源：[实际固定版本的 adapter_config.json](https://huggingface.co/Loie/SpotSound/blob/07eca9f048599fde92a55f2376e429deaa73c21a/adapter_config.json)，SHA-256 `67e69cec48b0217dce4109a9227ea176d4e3ee4689267fa058cd8960596d59b7`，与实验冻结记录完全匹配。
- 生成配置来源：[实际固定版本的 generation_config.json](https://huggingface.co/nvidia/audio-flamingo-3-hf/blob/7d4bae64ee29878af6504ae6f6bb3e40492838ad/generation_config.json)，SHA-256 `d6b329f8e2422195fda3ea6a5257035de7d973e3e6c2a664b86bbed6c03b367d`，与冻结记录完全匹配。其 2048-token 上限被实际评测脚本的 128 覆盖；单 beam/贪心含义已结合冻结运行环境 Transformers 5.9.0 的[生成配置代码](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/generation/configuration_utils.py)核实，未凭经验填写。
- 原实验仓库证据：`results/reltwin_review_controls_20260912/launch_snapshot/freeze.json`、`results/reltwin_paper_polish_20260913/bridge_complete/freeze.json` 及 `stress_complete/freeze.json`。训练脚本 `scripts/train_reltwin_micro.py` 的 SHA-256 为 `ec84bf3e7ad6585bf810ab44144512be9c05e55e7e00431519c2aa09ecf4309d`；评测脚本 `scripts/evaluate_checkpoint.py` 为 `e350b533365190735619da41b7f91d609ee70054fd9c7ca693041065cc5b5111`。
- 验收：配置已写入当前 LaTeX 与重新编译的 `paper.pdf`；五页编译通过，所有实验分数、图表及模型结果保持原样。

### C02 · 局部目标窗口的定义有歧义

- 状态：**待改。**
- 位置：Section 2.1，`Each spans the entire ordered sequence` 与 `Both queries are present at recording level`。
- 问题：前者容易被理解为两个目标都覆盖完整录音；后者把 query 和 query 所描述的事件关系混为一谈。
- 修改：明确每个目标窗口覆盖对应局部有序事件对的起止范围，而非整段录音；真正同时成立的是两种关系描述，其正确时间区间不同。可复用现有示意图说明。
- 验收：仅看本段即可知道两个窗口分别包含什么、为什么不能交换答案。无需补实验。
- 原审阅：#21、#22。

### C03 · Category-query rehearsal 的作用没有解释清楚

- 状态：**待改。**
- 位置：Section 2.1–2.2。
- 问题：正文提到为 A/B 的全部出现位置构造类别查询并加入 rehearsal，但没有直接解释保留类别定位监督与关系训练的联系。
- 修改：用一句话说明该项用于保留类别级定位监督，并说明 SFT 对照同样使用该项；不要未经消融便声称它独立提高了指标。
- 验收：任务、损失项和对照设置能相互对应。无需补实验。
- 原审阅：#23。

### C04 · 结果段还有“列出数字但没有解释用途”的内容

- 状态：**待改；优先利用已有逐行结果。**
- 位置：Section 4.1、Figure 1(c)。
- 现状：92/239/69 的胜平负、两条 IoU 下降至少 0.5 的样本，以及成功案例的边界欠覆盖均被列出，但退化样本的错误类型未说明；`25 remain incorrect` 又用失败数重复呈现成功率。
- 修改：从已有记录核实两条明显退化究竟涉及关系选错、区间边界还是其他错误，再决定是否用一句概括。围绕“从同一答案变为按查询选对窗口”解释 37/160 → 1/160、27 对恢复及 51 对改善；避免只罗列计数。
- 验收：给出的统计或案例能支撑一个明确发现；不凭猜测给失败归因，不改变图中真实时间戳、覆盖范围和 IoU。边界修正不恢复为本稿的方法支线。
- 实验需求：缓存分析即可；缺证据时标明待核，不启动 GPU 实验。
- 原审阅：#53、#55、#57、#59。

### C05 · 三种比较口径和置信区间的含义需更易读

- 状态：**表达待简化；数值与必要边界已保留。**
- 位置：Section 3、Table 1–2、Section 4.1/4.3。
- 当前正确口径：

| 比较用途 | 当前结果 | 可以支持什么 |
|---|---|---|
| 同环境、相同数据与更新数的 seed-0 SFT 对照 | SpotSound 58.42 → 59.43，+1.02；CI [0.23, 1.81] | 本次匹配运行中加入候选监督的收益 |
| RelTwin 三种子均值 vs 论文主表最佳 | 59.16 vs 57.9，+1.26 | 相对所引用既有主表的性能提升 |
| RelTwin 三种子均值 vs 本环境 official 复评 | 59.16 vs 58.42，+0.75；CI [−0.24, 1.74] | 正的点估计差；该区间尚不排除零差异 |

- 修改：让表注只解释该表的比较对象；在设置中集中说明 bootstrap 单位。公共集按音频，关系集另按原始来源连接组处理复用；解释区间基于已观察到的训练结果，不把它写成多训练种子的稳定性证明。
- 验收：摘要与正文不会把 +1.26 当成损失消融增益，也不会把 official 对照的跨零区间写成显著胜出。
- 原审阅：#9、#45、#50、#70、#72、#73、#84–#86。接受去重，不接受删除影响结论的统计事实。

### C06 · Timing intervention 的结论需要保持对象清晰

- 状态：**部分已解决；仍可精简实现细节、加强解释。**
- 位置：Section 4.2。
- 当前证据：80 段中 40 段波形未变、40 段改变时间布局；20.00 点的 JointPairAcc 优势来自发生改变的 40 段，不是把未改变样本也当成干预成功。对应原布局优势为 42.50 点。
- 修改：保留实际干预对象和冻结模型条件，把“逐条预测完全重现”的核验过程移出核心讨论。解释统一间隔明显帮助 SFT，但仍保留 RelTwin 的选择优势；不把该试验写成消除了全部捷径或验证自然场景泛化。
- 验收：段落以所支持的机制发现结束，读者能区分 40 段干预子集与全部 160 个查询的统计。
- 实验需求：无需补实验。
- 原审阅：#10、#18、#26、#65–#67、#89。

### C07 · 创新主张和证据范围需要守住，不是靠删除限定来扩大结论

- 状态：**保留边界；不是强制补实验清单。**
- 当前论文研究的是：在同一录音中构造两种都成立、但对应不同区间的逆序关系，利用局部时间戳答案竞争强化查询—窗口对应。核心结果来自 RelTwin 候选监督，不是已删除的动态边界模块或旧路由系统。
- 当前还没有的证据：相对其他 hard-negative 构造的独立优势；三组匹配 SFT/RelTwin 种子对照；多个独立自然关系数据集上的验证。当前稿的外部 benchmark 是 SpotSound，合成关系 development 不是第二个外部 benchmark。
- 需要保留的事实：关系开发集与公开分数参与过项目开发；adaptation/development 的来源隔离不等同于审计了基础模型的全部预训练数据；相同更新数不代表相同 FLOPs。
- 处理：集中在设置与简短 Scope 交代一次，其余段落主讲已证实的发现。当前无需因这些边界自动重开大实验；若未来要声称独立选择集泛化、全面优于其他负例方法或多种子稳健性，再单独批准相应对照。
- 验收：可以删重复辩护，不能把已知开发集写成未使用过的测试集，也不能将旧 Clotho 结果重新归于 RelTwin。
- 原审阅：#20、#37–#39、#45、#52、#67、#73、#81–#87。

## P2 · 用语与措辞（在内容准确的基础上润色）

### W01 · 标题未准确突出 query–window correspondence

- 状态：**待改。**
- 现状：`Learning Query-Specific Windows from Co-Occurring Inverse Relations` 中的 `learning windows from relations` 搭配仍生硬，容易被理解为生成窗口。
- 修改方向：把标题中心落在“query–window binding”或“query-specific localization”；保留 RelTwin 以及同录音逆关系竞争的特点。不在此清单中提前冻结新标题。
- 原审阅：#1。

### W02 · 摘要仍有名词堆叠和单句负担过重

- 状态：**待改；旧稿部分问题已解决。**
- 现状：`locally confusable answer negative` 仍不自然；窗口 `answers the opposite query` 可改为更明确的“对应另一查询的正确目标”。seed-0 比较的条件、三个指标增益仍集中在一个长句；`full SpotSound-Bench` 可直接表达为对 400 个查询的评估或删去冗余范围修饰。
- 修改：按问题、方法、主要发现组织；压缩公平性条件，不取消对照对象；摘要已没有旧稿的跨零说明堆叠和消极结尾，不要重复修这些旧问题。
- 原审阅：#2、#5、#7、#8、#12。

### W03 · 方法段还有生硬搭配与间接指标定义

- 状态：**待改。**
- 现状：`controls a constant-position preference`、`fixes source material` 以及 `Swap error means ... violates this strict preference` 不够直接。
- 修改：分别说明平衡哪种关系先出现、复用完全相同的音频片段以控制声学内容，以及至少一个预测对自身目标的 IoU 不大于对相反目标的 IoU 即计为 swap error（包括平分）。必要时用紧凑条件式。
- 验收：不把“控制变量”译成“修复素材”；指标定义不用回看上一句才能判断。
- 原审阅：#24、#25、#34；与 C02/C03 协同处理。

### W04 · 实验设置仍有内部记录式标题与过程描述

- 状态：**待改。**
- 现状：`Data and split roles`、行对齐/时长核验清单，以及 `Code, inputs, weights, and environment are hashed ...` 占用正文，却不直接解释科学方法。
- 修改：标题改为清楚的数据与评估协议表述；哈希、逐行检查和重现核验放代码或复现记录，正文保留必要协议及统计方法。`current` 仅在真正需要区别比较对象时使用。
- 参数精度：20,185,088 是准确的参数数量，不是因为结果保留两位小数就构成错误；为易读可写约 20.19M，但不能借此删除关键配置。
- 原审阅：#36、#43、#48、#49、#66、#92。

### W05 · 案例图叙述仍偏向缓存记录和残余错误

- 状态：**待改。**
- 位置：Figure 1 caption、panel (c)、Section 4.1。
- 现状：`cached seed-0 example`、`cached recovery case` 带内部工程色彩；`q+ still under-covers its target` 把成功案例的视觉重心落在残余错误上。
- 修改：去掉 `cached`，保留需要的模型身份；优先说明逆查询对应不同正确窗口的恢复，边界误差采用中性说明。保留原始条形范围、时间戳、IoU 和必要统计，不画成完美定位。
- 原审阅：#55–#57。

### W06 · 去除重复限定与固定“但不证明……”句式

- 状态：**部分已解决，需最后一轮通读。**
- 现状：v11 已显著减少 `not ...` 句式，很多原审阅引文已不存在；仍需把设置、表注、Scope 中相似说明集中，并避免重复用剩余失败数量解释成功率。
- 修改：每项边界放在最相关位置一次；结果段按“发现—证据—解释”组织。不能为消除“防御性”删除开发用途、比较口径、跨零 CI 或改写不支持的能力。
- 验收：不机械禁用 `not`/`without`；如推理时不需要候选属于真实方法特征，应保留。
- 原审阅：#12、#45、#50、#59、#73、#83–#86、#91、#93、#95、#96。

### W07 · 结论里的 advantage 仍过于笼统

- 状态：**待改。**
- 现状：`retaining an advantage under timing intervention` 没明确说明保留的是关系选择/联合定位优势。
- 修改：结论围绕“两个都真实存在的声音窗口仍需要按查询区分”收束，点明候选监督改善的能力及不改变推理流程的意义，不重复堆三套比较数字。
- 原审阅：#10、#89。旧稿以“尚未建立泛化”结尾的问题已解决。

## 原审阅 96 条意见的归类与处置索引

以下每个原编号只在本表归档一次。`已处理`表示问题引文已被改写、相关段落已删除或旧结构已取消，并不表示整节再无润色空间。`保留事实`表示不采纳通过删掉必要事实来增强宣传的处理方式。P0 项主要来自格式审计与作者信息核验，原 96 条主要覆盖 P1/P2。

| 原编号 | 类别 | v11 状态与合并后的处理 |
|---|---|---|
| 1 | P2 | 待改标题搭配，见 W01。 |
| 2 | P2 | 原句已变，仍有窗口“回答查询”的同类表达，见 W02。 |
| 3–4 | P2 | 已处理：摘要已改为直接说明构造和训练，不再使用原来的 where/relations identify answers。 |
| 5 | P2 | 部分保留名词堆叠，见 W02。 |
| 6 | P2 | 已处理：不再用 Standard 修饰目标；准确写 cross-entropy 本身仍应保留。 |
| 7–8 | P2 | 部分待改：摘要结果长句和范围修饰，见 W02。 |
| 9 | P1 | 摘要旧句已删除；统计分组在设置中有说明，继续按 C05 简化。 |
| 10 | P1/P2 | 摘要已明确 20.00 点及干预录音；结论对象仍可明确，见 C06、W07。 |
| 11 | P2 | 已处理：摘要不再以“泛化尚未建立”结束。 |
| 12 | P2 | 部分已解决；摘要压缩与全篇去重见 W02、W06。 |
| 13–16 | P2 | 已处理：相关搭配、Ordinary 修饰和 contribution 单复数已改写。 |
| 17–20 | P1/P2 | 引言旧表达已改写：贡献先行、给出具体结果、删除自我否定句；事实边界按 C07 保留。 |
| 21–22 | P1 | 待改：局部目标范围与事件关系定义，见 C02。 |
| 23 | P1 | 待改：rehearsal 作用说明，见 C03。 |
| 24–25 | P2 | 待改控制变量表述，见 W03。 |
| 26–29 | P1/P2 | 原负面句、内部版本定义与推理病句已改写；时间干预及推理输入说明保留，见 C06。 |
| 30–32 | P1/P2 | 随 RBEE/SetPO 扩展方法小节删除而关闭，不恢复该支线。 |
| 33 | P2 | 已处理：改为 Joint localization and relation selection。 |
| 34 | P2 | 待改直接定义 swap error，见 W03。 |
| 35 | P2 | 已处理：改为与 IoU 一起报告，不再采用替代/不替代的辩护句。 |
| 36 | P2 | 待改内部式小标题，见 W04。 |
| 37–39 | P1/P2 | 原防御式表述已重写；开发用途与来源隔离范围是必要事实，按 C07 保留。 |
| 40–42 | P1/P2 | 已处理主要结构问题：训练表已去掉历史 SFT、RBEE/SetPO 等非同协议支线；当前对照和多种子统计按 C05 区分。 |
| 43 | P2 | 可选润色；精确参数数目不是技术错误，可用 20.19M，见 W04。 |
| 44 | P2 | 已处理：原显然性说明不再出现。 |
| 45 | P1/P2 | 多处重复已减少；设置中等更新不等算力仍需保留一次，见 C05、C07、W06。 |
| 46–47 | P1/P2 | 历史环境和次级扩展对照随删节关闭；不把当前三种子统计误当三组匹配对照。 |
| 48–49 | P2 | 待处理内部核验清单与哈希句，见 W04。 |
| 50 | P1/P2 | 简化统计说明，不删区间条件，见 C05、W06。 |
| 51–52 | P1/P2 | 小节标题已改为正向发现；替代负例未比较集中在 Scope，不在结果后重复否定，见 C07。 |
| 53 | P1 | 两条明显退化尚缺错误类型解释，见 C04。 |
| 54 | P2 | 已处理：aggregate/not per-example dominance 原句已删除。 |
| 55–57 | P1/P2 | 待改缓存措辞与案例叙事，真实边界误差不可抹去，见 C04、W05。 |
| 58 | P2 | 已处理：不再使用 not only recovered cases；保留统计总体说明。 |
| 59 | P1/P2 | 仍有剩余失败数的重复表达，见 C04、W06。 |
| 60–64 | P1/P2 | 随 exchange/setwise 扩展结果及相关复杂对照删除而关闭。 |
| 65 | P2 | 已处理：标题改为 Binding gains persist after timing intervention。 |
| 66–67 | P1/P2 | 精简逐条重现的实现细节；保留 40/40 干预区别与开发后测试身份，见 C06、W04。 |
| 68–72 | P1/P2 | 主表已去除旧系统与无关列，改用正向来源说明；最高列值图例和文献/消融区别按 C05 保留。 |
| 73 | P1/P2 | 接受去重，不接受删去 official 对照跨零 CI，见 C05、W06。 |
| 74–80 | P1/P2 | 已处理：旧 routing/refinement、Clotho 支线、六位小数和 UnAV 协议进度句均已移出当前稿。 |
| 81–87 | P1/P2 | Scope 已缩短、重写；保留数据开发用途与单匹配种子等必要范围，见 C05、C07、W06，不以润色伪造更强证据。 |
| 88 | P2 | 已处理：结论以方法贡献起句，不以 seed-0 公平性条件起句。 |
| 89 | P2 | 仍需明确结论中 advantage 的对象，见 W07。 |
| 90 | P2 | 已处理：结论不再以未建立泛化收尾。 |
| 91–93 | P2 | 部分已解决；内部词、重复限定及句式通读见 W04、W06。 |
| 94 | P1 | 已处理：全文聚焦 RelTwin，旧系统和扩展支线已移除。 |
| 95–96 | P2 | 大部分旧句已改写；剩余段落按发现—证据—解释整理，见 W06。 |

## 当前稿不得混淆的已完成调整

- 论文统一称 **RelTwin**，对应此前 RelTwin-Cand 的 `no_exchange` 配置，不是给旧完整系统换名字。
- RBEE、SetPO、旧路由/边界系统及其结果不再出现在当前方法与主表中；历史实验记录没有删除。
- 当前主表是 RelTwin 对所引用的 SpotSound 论文主表各方法，提升以此前主表最佳为参照；不是用旧 main 的 59.33 来充当外部榜首。
- Clotho 的 86.86 属于旧流程，不能作为当前 RelTwin 的跨 benchmark 成绩重新写入。
- 稿件现有 matched SFT 对照和诊断结果应保留；不能把所有对照一概当作“中间方法”删掉。
- 核心现象是查询—窗口对应错误，不把不到 0.1 点的旧边界修正收益重新包装成本稿核心创新。

## 维护约定与版本记录

本分支同时维护 LaTeX 源码、`paper.pdf` 与 `ISSUES.md`，协作者可直接编辑并编译 `main.tex`。论文的固定阅读入口仍为 PDF 与本清单，编译说明见 `README.md`。实验记录与训练代码保留在原开发分支；`paper` 分支聚焦论文协作。

每轮先从 `paper` 建新的修改分支，完成后提交 Git，再以正常快进或合并更新 `paper`，不强推覆盖历史。修改论文时同步提交源码、重新编译的 `paper.pdf`，以及本文件的版本、哈希和问题状态；不要再增加 `final2`、`v12-final` 等平行文件。关闭事项写明验证结果，新增问题归入 P0/P1/P2。只有问题清单变化时，注明 PDF 未变。

| 日期 | PDF 版本 | 本次变化 |
|---|---|---|
| 2026-09-14 | v11，内容未变 | 建立 `paper` 文档入口；合并当前审计与 96 条外部审阅意见，区分待办、已处理与必须保留的事实。 |
| 2026-09-14 | v11，内容未变 | 按作者要求将 PDF 固定命名为 `paper.pdf`，同步更新清单链接；仅改文件名，PDF 哈希不变。 |
| 2026-09-14 | v11，内容未变 | 补入 19 个 LaTeX/图表/文献/样式文件与协作编译说明；独立编译的五页 PDF 与当前稿文字、渲染像素一致，单独主图也编译通过。从双文件交付扩展为可编辑论文分支，现有问题状态不变。 |
| 2026-09-14 | v12，C01 配置补充 | 核查冻结配置与源码，补齐 LoRA、每步样本/梯度累积和解码设置；C01 关闭，重新编译并检查 PDF。未训练、未重新推理，实验数值未改，其余待办保留。 |
| 2026-09-14 | v12，Figure 1 合并稿 | 合入主图重绘：撤去结果流向图以及 Figure 1 底部 inference pipeline 和整体统计；(c) 改为 SFT／RelTwin 两列的查询—窗口对照，三幅子图统一以绿／金／紫替代大面积灰底。时间戳、IoU 与实验数值未改。 |
| 2026-09-14 | v12，浅色分区与深黑文字稿 | 按作者提供的 SplineGS 配色参考，改为浅杏／淡紫／浅绿分区底；浅绿和淡紫对应两类查询及窗口，所有英文、公式、符号和 IoU 统一深黑，重点保留加粗。修正训练框顶部和输入符号的间距。独立图、灰度图、论文尺寸已检查；五页论文逐页词项、数据宏和配置说明均未改变。 |
| 2026-09-14 | v12，功能配色与前景层次稿 | 根据作者关于背景与模块重复、色彩单一的反馈，撤去三块整栏底色，采用浅暖底与局部功能色区：青色音频／模型、紫色监督区、杏色正确候选，C 图绿／紫窗口加深并使用对称的模型标题条。英文和公式继续统一深黑。独立图、灰度图、论文内尺寸检查通过；五页词项与实验数据未改变。 |

### 可追溯信息

- v11 源起点：`3841a85d00ed4487be2cca7e1022b2279d079d79`。当前 v12 的 PDF 由本次提交内的 LaTeX 源生成，以包含 v12 版本记录的 Git 提交为准。
- 原交付审计提交：`a190b11`；原开发分支：`codex/reltwin-core-only-20260913`。
- 原 LaTeX 路径：`paper/overleaf_icassp2027_reltwin_final_v11`；对应的编译源文件已纳入本分支，入口为 `main.tex`。v12 已补充 `sections/experiments.tex` 并更新 Figure 1 源码及导出文件。
- 当前 PDF SHA-256：`360c5d0a313e7059b8af988931a5abcc4035f387e720d087612b4e8f16e7a172`。
- 原审阅文件 SHA-256：`580a6e5edbd61812f68e1bf1c7d1157579a7050d88ad5e8394fc667526aff3c4`。
- 官方规则核验日期：2026-09-14。后续若规则更新，重新核查 F01–F06，不能仅凭本清单确认可投稿。
