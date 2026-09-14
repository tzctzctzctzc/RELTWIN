# RelTwin · ICASSP 2027 论文协作

[当前论文 PDF](paper.pdf) · [问题清单](ISSUES.md) · [LaTeX 主文件](main.tex)

`paper` 分支同时保存可编辑源码和当前编译稿。当前 v12 审阅稿已补齐已有实验的 LoRA、每步样本/梯度累积及解码设置，并同步更新 Figure 1：删除底部 inference pipeline 与整体统计，右栏改为 SFT／RelTwin 的查询—窗口对照；三幅子图统一减少灰底。没有训练、重新推理或改动实验结果。待处理问题、作者信息占位及提交前检查统一记录在 `ISSUES.md`。

Figure 1 参考作者提供的 SplineGS 配色图，采用青色 `#BFDCE7`／`#DFEDF2`、淡紫 `#CAC2D7`／`#DDD6E5`、浅绿 `#D9E4C2` 和浅杏 `#F5D8BF`。整图使用浅暖底 `#FBF8F3`，颜色集中于功能区域：青色标音频／模型，紫色组织候选监督，杏色强调正确候选。两类查询用浅绿／淡紫，C 图窗口采用加深的同色填充 `#A8BF8C`／`#B5A3C9`，两个模型使用相同的浅青标题条。全部英文、数学符号和数值保持深黑 `#242424`。按画图 skill 的导出检查方式核对独立图、灰度图和论文内尺寸；原始时间戳与四个 IoU 未改，PDF 中所有主图文字均已核验为同一色值。

## 修改哪里

| 文件或目录 | 用途 |
|---|---|
| `main.tex` | 论文入口、宏包与标题 |
| `authors.tex` | 作者、单位与通讯作者信息 |
| `sections/` | 摘要、引言、方法、实验、讨论、结论与声明 |
| `tables/` | 主表和训练对照表 |
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

每轮从最新 `paper` 分支建立新的修改分支，提交后通过正常快进或合并更新 `paper`，不强推覆盖历史。论文改动应同步提交 **LaTeX 源码、重新编译的 `paper.pdf` 和更新后的 `ISSUES.md`**。只改问题清单时，明确注明 PDF 未变。

不新增多个 `final`/版本号 PDF，也不将 `.aux`、`.log` 等编译缓存提交。问题关闭前应核验对应 PDF；数值、图中时间戳或实验结论的变化须有对应实验依据。

当前源文件来自冻结提交 `3841a85d00ed4487be2cca7e1022b2279d079d79` 的 `paper/overleaf_icassp2027_reltwin_final_v11`。初次补入的 19 个 LaTeX/图表/文献/样式文件保持原始内容；后续以本分支提交历史追踪修改。`paper.pdf` 的版本及 SHA-256 见 `ISSUES.md`。

2026-09-14 初次补入 v11 源码时，19 个源文件与冻结提交一致；独立编译的五页 PDF 与当时的 `paper.pdf` 文字及像素一致，独立主图也编译通过。随后 v12 补充 `sections/experiments.tex` 的配置说明并合入 Figure 1 更新，重新编译后仍为五页，无 overfull box 或未解析引用。C01 已关闭；其余问题不因编译通过而自动关闭，具体状态与当前 PDF 哈希见 `ISSUES.md`。
