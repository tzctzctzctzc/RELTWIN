# RelTwin · ICASSP 2027 论文协作

[当前论文 PDF](paper.pdf) · [问题清单](ISSUES.md) · [LaTeX 主文件](main.tex)

本分支保存可编辑源码和当前编译稿。当前为 v16（表格与分析审阅稿），基于 `paper` 分支 v15，在 `docs/reltwin-table-analysis-20260914` 上修改，供作者确认后再合入 `paper`。标题为 **RelTwin: Contrasting Locally Valid Timestamp Answers for Query-Specific Audio Grounding**。Table 1 恢复公开方法分组，Table 2 跨栏呈现基础模型、匹配训练对照及三种子汇总，并加入已有的 Swap error 和同答案对数。结果文字按绑定恢复、时间干预、公开性能展开；比较口径集中说明。没有训练、重新推理或改动预测。正文至第 4 页结束，第 5 页为声明和参考文献。待处理问题、作者信息占位及提交前检查统一记录在 `ISSUES.md`。

Figure 1 参考作者提供的 SplineGS 配色图，采用青色 `#BFDCE7`／`#DFEDF2`、淡紫 `#CAC2D7`／`#DDD6E5`、浅绿 `#D9E4C2` 和浅杏 `#F5D8BF`。整图使用浅暖底 `#FBF8F3`，颜色集中于功能区域：青色标音频／模型，紫色组织候选监督，杏色强调正确候选。两类查询用浅绿／淡紫，C 图窗口采用加深的同色填充 `#A8BF8C`／`#B5A3C9`，两个模型使用相同的浅青标题条。全部英文、数学符号和数值保持深黑 `#242424`。按画图 skill 的导出检查方式核对独立图、灰度图和论文内尺寸；原始时间戳与四个 IoU 未改，PDF 中所有主图文字均已核验为同一色值。

## 修改哪里

| 文件或目录 | 用途 |
|---|---|
| `main.tex` | 论文入口、宏包与标题 |
| `authors.tex` | 作者、单位与通讯作者信息 |
| `sections/` | 摘要、引言、相关工作、方法、实验、讨论、结论与声明 |
| `sections/related_work.tex` | 独立 Related Work 章节及与最近邻方法的结构比较 |
| `tables/main_results.tex` | Table 1：分组的 SpotSound-Bench 公开方法比较 |
| `tables/training_results.tex` | Table 2：跨栏的匹配训练、关系绑定诊断与三种子汇总 |
| `figures/overview_art.tex` | 可编辑的 TikZ 主图 |
| `figures/overview_data.tex` | 主图使用的时间戳与统计值 |
| `figures/reltwin_overview.tex` | 主图排版和图注 |
| `overview_figure.tex` | 单独编译主图的入口，可选 |
| `references.bib` | 参考文献数据库 |
| `spconf.sty`、`IEEEbib.bst` | 会议排版与参考文献样式 |
| `paper.pdf` | 固定名称的当前论文 PDF |
| `ISSUES.md` | 按优先级维护的问题、处理状态和版本记录 |

源码直接放在分支根目录，所有编译依赖都使用相对路径，不需要实验仓库、数据集、模型权重或 GPU。历史实验审计文件不参与编译，也没有复制为新的论文内容。

## Overleaf

在 GitHub 切换到本次审阅分支 `docs/reltwin-table-analysis-20260914`（已接受的协作稿仍位于 `paper`），使用 **Code → Download ZIP**，再在 Overleaf 中选择 **New Project → Upload Project**。也可以将已解压的本分支文件直接上传。

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

每轮从最新 `paper` 分支建立新的修改分支，提交后通过正常快进或合并更新 `paper`，不强推覆盖历史。论文改动应同步提交 **LaTeX 源码、重新编译的 `paper.pdf` 和更新后的 `ISSUES.md`**。只改问题清单时，明确注明 PDF 未变。

不新增多个 `final`/版本号 PDF，也不将 `.aux`、`.log` 等编译缓存提交。问题关闭前应核验对应 PDF；数值、图中时间戳或实验结论的变化须有对应实验依据。

当前源文件来自冻结提交 `3841a85d00ed4487be2cca7e1022b2279d079d79` 的 `paper/overleaf_icassp2027_reltwin_final_v11`。初次补入的 19 个 LaTeX/图表/文献/样式文件保持原始内容；后续以本分支提交历史追踪修改。`paper.pdf` 的版本及 SHA-256 见 `ISSUES.md`。

2026-09-14 初次补入 v11 源码时，19 个源文件与冻结提交一致；独立编译的五页 PDF 与当时的 `paper.pdf` 文字及像素一致，独立主图也编译通过。随后 v12 补充 `sections/experiments.tex` 的配置说明并合入 Figure 1 更新，重新编译后仍为五页，无 overfull box 或未解析引用。C01 已关闭；其余问题不因编译通过而自动关闭，具体状态与当前 PDF 哈希见 `ISSUES.md`。

v13 仅关闭 P0-01：摘要为 130 词，新标题按原模板排为两行；重新编译仍为五页，无 overfull box 或未解析引用。相关工作、方法、训练配置、结果段、图表与文献未改。关于声学有效性、SFT 对比含义和结果口径的修改后审查已记录在 `ISSUES.md` 第 7 节。

v14 继续精炼 P0-01：摘要为 122 词，保留关系绑定提升和 SpotSound 公共结果两项核心证据；贡献段不再重复实验数字。种子和配对设置、时间干预数值及完整结果仍可在实验部分核对。标题、引言前四段、结论、方法、实验、图表和参考文献均未修改。重新编译仍为五页，无 overfull box 或未解析引用；其他问题状态保持不变。

v15 关闭 P0-02：新增约 280 词的独立 Related Work，原引言仅保留问题、方法概述与贡献。正文按监督对象区分表示对齐、局部显著性排序和完整时间戳答案比较；原文核对记录见 `ISSUES.md` 第 7 节。Figure 1 保留在第 2 页，引用与章节交叉引用自动更新；仍为五页、17 条参考文献，无 overfull box 或未解析引用。正文充实与参考文献独占第五页的最终排版仍由 P1-08 跟进，本轮没有将其标为完成。

v16 重整两张表并补充分析：公开主表保留十个基线和全部原始分数，恢复四组方法类别、最优/次优标记及指标方向；跨栏机制表补入同一批既有预测的错误指标和三种子统计。P1-02、P1-05 已完成，其他相关条目记录进展，不因本轮排版通过而一并关闭。逐行核对原始 evidence/figure evidence 后编译为五页，Table 1 在第 3 页，Table 2 在第 4 页，第 5 页仅声明与 17 条参考文献；无 overfull box、未解析引用或 Type 3 字体。作者、摘要、引言、Related Work、方法、主图、结论、声明及参考文献源码均未改。
