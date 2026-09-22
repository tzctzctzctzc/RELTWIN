# RelTwin · ICASSP 2027 论文协作

[当前论文 PDF](paper.pdf) · [问题清单与故事线](ISSUES.md) · [LaTeX 主文件](main.tex)

当前为 **v34（2026-09-22，补充 Introduction 与结果分析）**。标题为 **RelTwin: Contrasting Locally Valid Timestamp Answers for Query-Specific Audio Grounding**。张跃瀚（Yuehan Zhang）与汤智铖（Zhicheng Tang）为共同第一作者，张跃瀚列第一、汤智铖列第二；Wei Xu 为通讯作者。全文只围绕“两个窗口都有真实声学支持，但查询决定哪一个是答案”组织。Method 按局部有效答案构造、查询条件竞争、集合级偏好学习展开；辅助类别查询配对仅在训练设置交代，不命名为新模块或贡献。Introduction 新增评估职责说明，结果部分补充公开迁移、配对机制和 timing intervention 的关系分析，未增加实验或改变数据。

本轮沿用 v30 的基线出处、51:1 配对转变和时间干预结果整理；正文暂不展开模型选择过程，供内部讨论使用。两张表的数值和置信区间不变。

Table 1 包含 SpotSound-Bench、Clotho-Moment、UnAV-100 subset 三组，每组按 R1@.3、R1@.5、mIoU 排列，分组横线和 12 pt 额外列间距区分 benchmark。TimeAudio / TimeAudio + RelTwin、SpotSound-A / SpotSound-A + RelTwin 分别相邻排列，各有相对对应原文基础模型的增益行。TimeAudio 获得 **18.36 / 34.08 / 30.48 mIoU**，增益 **+8.16 / +5.48 / +14.48 点**；SpotSound-A 为 **59.43 / 86.68 / 74.03 mIoU**，增益 **+1.53 / +1.08 / +4.23 点**。所有数字与 v24 相同，删除 † 及版本区分表注。作者本轮确认 SpotSound-A 这组数值来自同事在另一机器上的完整 RelTwin 实验，按此确认保留；其原始记录待归档（P0-10），不沿用旧 `no_exchange` 日志充当本次完整实验依据。

Table 2、逆查询诊断和 timing 仍承担候选监督的机制验证，标签明确为 candidate supervision，既有数值和 CI 不变。结果节统一为公开定位表现、关系选择机制、时间干预三层，删去重复读表和独立的重复解释小节。开发用途、五次运行选择与三种子统计保留于设置；五种子汇总仍由 P3-02 跟踪。修订未运行训练或推理，也未补造完整 SpotSound-A 的缺失参数。

当前正文引用 **19 篇真实文献**，真实性、发表元数据与引用用途核对记录见 `ISSUES.md` 第 7 节 v29。将 Text-to-Audio Grounding、Audio Moment Retrieval、T-CLAP 和弱监督 TAG 更新为核验过的正式发表版本；补入 ICASSP 2024 反事实音频学习近邻和 TEMPO 训练数据来源。未引用的五条历史条目已从当前文献库清理，可由 Git 恢复。正文四页、第五页声明与参考文献，固定文件仍为 `paper.pdf`。

Figure 1 按作者提供的配色参考重绘：上方为共享音频与逆查询，下方蓝色 RelTwin 框从左到右展示候选竞争与交换一致性 → 集合级偏好学习；绿色推理条单独成行，右侧橙色面板展示候选监督的真实预测对照。采用白底细边框、深蓝与橙色悬浮标题、浅绿与浅杏色高亮；无衬线字体统一为 8 pt 正文和 9 pt 标题。案例端点和 IoU 不变。唯一图稿来源为 `figures/overview_art.tex`；`previews/nova_overview.*` 为历史命名的兼容入口，与论文图同步。

v33 在 Introduction 末尾补充评估职责说明，并在第 4 页 Results 的 timing intervention 后补充一段短分析：明确公开 benchmark、配对关系测试和静音布局干预分别承担迁移、机制和位置泛化证据；同时说明集合级偏好学习覆盖 coverage、event count 与 boundaries。正文仍在第 4 页结束，第 5 页仅声明与参考文献。

## 修改哪里

| 文件或目录 | 用途 |
|---|---|
| `main.tex` | 论文入口、宏包与标题 |
| `authors.tex` | 作者、单位与通讯作者信息 |
| `sections/` | 摘要、引言、相关工作、方法、实验、讨论、结论与声明 |
| `sections/related_work.tex` | 独立 Related Work 章节及与最近邻方法的结构比较 |
| `sections/method.tex` | 局部目标构造、查询—答案得分矩阵与两阶段训练目标 |
| `sections/experiments.tex` | 数据、成对指标、训练与评测设置、比较及统计口径 |
| `tables/main_results.tex` | Table 1：三个 benchmark 的基础模型与 RelTwin 附加微调结果，差值相对对应原文基础模型 |
| `tables/training_results.tex` | Table 2：跨栏的匹配训练、关系绑定诊断与三种子汇总 |
| `figures/overview_art.tex` | 可编辑的 TikZ 主图 |
| `figures/overview_data.tex` | 主图使用的时间戳与统计值 |
| `figures/reltwin_overview.tex` | 主图排版和图注 |
| `overview_figure.tex` | 单独编译主图的入口，可选 |
| `previews/nova_overview.tex`、`previews/nova_overview_art.tex` | 历史命名的兼容入口，现在引用 RelTwin 图稿 `figures/overview_art.tex` |
| `previews/nova_overview.pdf`、`previews/nova_overview.png` | 当前 Figure 1 的矢量 PDF 和图片预览，与论文共用图稿 |
| `references.bib` | 参考文献数据库 |
| `spconf.sty`、`IEEEbib.bst` | 会议排版与参考文献样式 |
| `paper.pdf` | 固定名称的当前论文 PDF |
| `ISSUES.md` | 按优先级维护的问题、处理状态和版本记录 |

源码直接放在分支根目录，所有编译依赖都使用相对路径，不需要实验仓库、数据集、模型权重或 GPU。历史实验审计文件不参与编译，也没有复制为新的论文内容。

## Overleaf

在 GitHub 切换到 `paper` 分支，使用 **Code → Download ZIP**，再在 Overleaf 中选择 **New Project → Upload Project**。也可以将已解压的本分支文件直接上传。

将 **Main document** 设为 `main.tex`，编译器选 **pdfLaTeX**。参考文献使用 **BibTeX**，Overleaf 的自动编译会处理。不要将 `overview_figure.tex` 误设为整篇论文入口。

`paper.pdf` 是版本化的阅读稿，不是 LaTeX 输入。Overleaf 修改后请同步源文件，并将编译出的 PDF 更新为仓库中的 `paper.pdf`。

## 本地编译

安装包含 TikZ、booktabs、hyperref 等常用宏包的 TeX Live 或 MiKTeX。在本分支根目录执行：

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

若已安装 latexmk，也可以执行：

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

输出为 `main.pdf`。核对页数、图表、引用及作者信息后，将其复制为 `paper.pdf`。Windows PowerShell 可使用：

```powershell
Copy-Item -LiteralPath main.pdf -Destination paper.pdf
```

Linux/macOS 可使用 `cp main.pdf paper.pdf`。中间文件和临时编译 PDF 已由 `.gitignore` 排除，`paper.pdf` 正常纳入版本管理。单独编译主图时执行 `pdflatex overview_figure.tex`；该可选入口另需 `standalone` 宏包。

## 协作约定

按作者 2026-09-14 的最新要求，后续论文修改直接在 `paper` 分支维护并提交，不再停留在独立审阅分支等待合并确认。修改前先获取远端更新，保留协作者修改；存在分歧时正常合并，逐项解决冲突，核对源码并重新编译 PDF 后推送，不强推覆盖历史。若冲突涉及无法由现有证据确定的实验数值或作者决定，先核实，不猜测或整文件覆盖。论文改动应同步提交 **LaTeX 源码、重新编译的 `paper.pdf` 和更新后的 `ISSUES.md`**。只改问题清单或协作说明时，明确注明 PDF 未变。

不新增多个 `final`/版本号 PDF，也不将 `.aux`、`.log` 等编译缓存提交。问题关闭前应核验对应 PDF；数值、图中时间戳或实验结论的变化须有对应实验依据。

当前源文件来自冻结提交 `3841a85d00ed4487be2cca7e1022b2279d079d79` 的 `paper/overleaf_icassp2027_reltwin_final_v11`。初次补入的 19 个 LaTeX/图表/文献/样式文件保持原始内容；后续以本分支提交历史追踪修改。`paper.pdf` 的版本及 SHA-256 见 `ISSUES.md`。

2026-09-14 初次补入 v11 源码时，19 个源文件与冻结提交一致；独立编译的五页 PDF 与当时的 `paper.pdf` 文字及像素一致，独立主图也编译通过。随后 v12 补充 `sections/experiments.tex` 的配置说明并合入 Figure 1 更新，重新编译后仍为五页，无 overfull box 或未解析引用。C01 已关闭；其余问题不因编译通过而自动关闭，具体状态与当前 PDF 哈希见 `ISSUES.md`。

v13 仅关闭 P0-01：摘要为 130 词，新标题按原模板排为两行；重新编译仍为五页，无 overfull box 或未解析引用。相关工作、方法、训练配置、结果段、图表与文献未改。关于声学有效性、SFT 对比含义和结果口径的修改后审查已记录在 `ISSUES.md` 第 7 节。

v14 继续精炼 P0-01：摘要为 122 词，保留关系绑定提升和 SpotSound 公共结果两项核心证据；贡献段不再重复实验数字。种子和配对设置、时间干预数值及完整结果仍可在实验部分核对。标题、引言前四段、结论、方法、实验、图表和参考文献均未修改。重新编译仍为五页，无 overfull box 或未解析引用；其他问题状态保持不变。

v15 关闭 P0-02：新增约 280 词的独立 Related Work，原引言仅保留问题、方法概述与贡献。正文按监督对象区分表示对齐、局部显著性排序和完整时间戳答案比较；原文核对记录见 `ISSUES.md` 第 7 节。Figure 1 保留在第 2 页，引用与章节交叉引用自动更新；仍为五页、17 条参考文献，无 overfull box 或未解析引用。正文充实与参考文献独占第五页的最终排版仍由 P1-08 跟进，本轮没有将其标为完成。

v16 重整两张表并补充分析：公开主表保留十个基线和全部原始分数，恢复四组方法类别、最优/次优标记及指标方向；跨栏机制表补入同一批既有预测的错误指标和三种子统计。P1-02、P1-05 已完成，其他相关条目记录进展，不因本轮排版通过而一并关闭。逐行核对原始 evidence/figure evidence 后编译为五页，Table 1 在第 3 页，Table 2 在第 4 页，第 5 页仅声明与 17 条参考文献；无 overfull box、未解析引用或 Type 3 字体。作者、摘要、引言、Related Work、方法、主图、结论、声明及参考文献源码均未改。

v17 仅完成 P0-03 和 P0-04：方法节将正答案序列监督与同录音答案间的显式相对偏好分开，补齐局部区间、三位小数序列化、监督 token 范围、2×2 得分矩阵、独立抽取的类别 rehearsal 及各项损失。成对指标直接使用 own-target/cross-target IoU 条件，保留平分计错和完全相同区间列表的规则。核对冻结训练代码哈希，复算 1,024 条既有训练日志损失及五个模型的 1,600 条缓存预测，并通过 150 个小矩阵数学检查；没有执行模型训练或推理。仅移动 `sections/bridge_results.tex` 中的一行表格接入指令以维持 Table 2 在第 4 页，结果文字和表格数值逐字保留。编译仍为五页，正文四页，第五页声明与文献；无 overfull box 或未解析引用，其他问题状态不变。

v18 仅关闭 P0-06：删除正文中的逐行核验、哈希说明和重复的公平性清单；完整训练匹配条件留在设置，训练终点评估明确为 256 次更新后。图注改为描述输入、监督与恢复现象，内部样本身份保留在 `ISSUES.md` 第 7 节。核对四次训练摘要及主图四个输入文件哈希，确认配置和图例数据未变。方法、两张表、结果数值及 CI、timing、Scope、Conclusion、作者、声明和文献不变。结论后增加标准分页，让声明与全部文献从第 5 页开始；五页均已渲染检查，所有字体嵌入，无 overfull box 或未解析引用。P1-01/P1-03 记录关联进展并转为待终检，其余未完成项继续保留。

2026-09-14 经作者确认，将 v16–v18 的三个提交 `0452f37`、`14aa630`、`094566e` 快进合入 `paper`，无冲突、不覆盖既有历史。本次合并只另行更新协作入口和台账状态；LaTeX 与 `paper.pdf` 保持 v18 原样，不新增论文版本号，也不改变问题处理状态。

v19 恢复截图的三 benchmark、十个基线、九列指标、此前逐项最佳和差值行。NOVA 系统结果与 RelTwin 独立对照分别呈现；摘要、引言、设置和结果同步调整。精炼重复解释以容纳系统配置和跨栏主表，保留核心方法公式、全部机制表数值、转移数量、CI 和 timing 数值。作者、主图、核心方法、机制表及会议样式未改。公开数据声明补齐恢复的数据集，启用已有 Auto-AEG 参考条目，伦理声明保留为具名段落。正文四页，第五页为声明与全部文献；未执行新模型实验。输入依据及未解决问题见 `ISSUES.md`。

v20 改为直接的成果与方法叙事：Results 先呈现主表提升，再解释 RelTwin 的窗口选择恢复和时间干预；摘要、引言、结论同步。删除结果段反复解释比较用途的句子，将初始预测分数集中到实验设置。主表仅改图注，全部数值、核心方法、机制表、图、统计区间及训练配置保持原值。没有新增实验或改变任何活动问题的完成状态。

2026-09-14 经作者确认，将 v19/v20 的 `25e5c45`、`e209eb1` 无冲突快进合入 `paper`。本轮只同步协作入口与版本记录，LaTeX 和 `paper.pdf` 保持 v20 原样，所有活动问题状态不变。

v21 清理防御性论述：引言明确 NOVA 三阶段的作用，结果中的未做实验清单改为机制讨论，图例分析直接说明查询对应窗口的恢复，时间干预突出改变布局后的 20 点优势。结论落在“监督哪个真实事件实例回答查询”。必要的配置、协议、CI、全部正负结果和数据开发用途保持可核对；两张表、方法公式、主图及作者信息未改。P1-06、P1-07 已完成，其他问题只记录关联进展。编译仍为四页正文加第五页声明与参考文献，全部字体嵌入，无 overfull 或未解析引用。

2026-09-14 经作者确认，将 `7f4b800` 无冲突快进合入 `paper`，同步协作入口和台账，LaTeX 与 `paper.pdf` 保持 v21 原样。后续改稿改为直接维护 `paper`，具体规则同时写入本仓库 `AGENTS.md`；此约定仅针对论文仓库，不改变其他实验仓库的分支管理规则。

v23 回到 v18 的 RelTwin 单方法主线，接入经核实的三 benchmark 独立推理结果，保留 TimeAudio 占位并添加对应基础模型增益。同步恢复 RelTwin 标题、方法图、完整方法与相关工作，移出 NOVA 系统组成；保留最新作者信息及三种子/匹配统计。所有改动在现有 `paper` 历史上提交，不重置到旧提交；具体来源、故事建议和剩余问题见 `ISSUES.md`。

v24 填入 TimeAudio 完整配置的三 benchmark 全量结果及九项增益，逐行核对 400/6649/100 条和同一 checkpoint。方法节将答案序列化按骨干说明，设置补充两阶段训练与 replay 来源；压缩重复表述保持四页正文。原 SpotSound-A 行、Table 2、主图、作者和全部既有统计保持不变。SpotSound-A 完整配置留待后续实验，当前稿先供导师审阅。

v25 根据作者对同事完整 SpotSound-A 实验的最新确认保留原主表全部数据，统一完整 RelTwin 叙事，删除 † 和 core/full 区分。方法正式写入 JS、SetPO 及 replay；主图同步训练顺序；已核验的旧候选监督记录继续作为 Table 2 机制证据。原始同事实验追溯另由 P0-10 归档，不把数值相同当作检查点相同。段落结构保持，相关工作仅压缩重复句以容纳方法定义。

v26 仅重绘主图并同步图注、单图预览与论文 PDF：训练阶段以实线箭头连接，推理路径与诊断面板明确分开。保留所有实际时间戳、IoU、两张表和正文源码。检查单图、双栏嵌入、字体嵌入和页数；论文仍为四页正文，第五页声明与参考文献，无 overfull box 或未解析引用。未运行新实验。
