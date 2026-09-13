# RelTwin · ICASSP 2027 论文协作

[当前论文 PDF](paper.pdf) · [问题清单](ISSUES.md) · [LaTeX 主文件](main.tex)

本论文工作分支同时保存可编辑源码和当前编译稿。当前为 v11 主图精简与配色更新稿：撤去结果流向图，Figure 1 删去底部 inference pipeline 和 160 对整体统计；Figure 1(c) 改为 SFT／RelTwin 两列、逆查询两行的紧凑对照，每格叠加目标轮廓和预测填充。图注和 PDF 已同步。待处理问题、作者信息占位及提交前检查统一记录在 `ISSUES.md`。

本轮从 [BLIP-2（ICML 2023），Fig. 2](https://proceedings.mlr.press/v202/li23q/li23q.pdf) 的矢量 PDF 提取浅蓝 `#B3C7E7`、浅珊瑚 `#FCB19C`、紫色 `#B178C2`；[InstructBLIP（NeurIPS 2023），Fig. 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a6a435e75419a836fe47ab6793623e6-Paper-Conference.pdf) 使用相同蓝色、珊瑚色和近似紫色。查询与预测窗口使用对应色，候选监督使用淡紫，文字／边框采用压暗后的同色系。原始时间戳、IoU 数据宏未改；原生 TikZ、独立 PDF/SVG/PNG 与五页论文已同步。撤下的结果图可以从 `da246f5` 恢复。

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
| `figures/reltwin_overview.pdf`、`.svg`、`.png` | 从 TikZ 导出的独立矢量图与预览；可编辑源仍为 `overview_art.tex` |
| `overview_figure.tex` | 单独编译主图的入口，可选 |
| `references.bib` | 参考文献数据库 |
| `spconf.sty`、`IEEEbib.bst` | 会议排版与参考文献样式 |
| `paper.pdf` | 固定名称的当前论文 PDF |
| `ISSUES.md` | 按优先级维护的问题、处理状态和版本记录 |

源码直接放在分支根目录，所有编译依赖都使用相对路径，不需要实验仓库、数据集、模型权重或 GPU。历史实验审计文件不参与编译，也没有复制为新的论文内容。

## Overleaf

本轮图稿在 `codex/reltwin-figure1-redesign`，尚未合入 `paper`。在 GitHub 切换到该修改分支，使用 **Code → Download ZIP**，再在 Overleaf 中选择 **New Project → Upload Project**；合并后可改用 `paper`。也可以将已解压的本分支文件直接上传。

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

初始源文件来自冻结提交 `3841a85d00ed4487be2cca7e1022b2279d079d79` 的 `paper/overleaf_icassp2027_reltwin_final_v11`。初次补入的 19 个 LaTeX/图表/文献/样式文件保留原始内容；后续改动以本分支提交历史追踪。`paper.pdf` 的版本及 SHA-256 见 `ISSUES.md`。

2026-09-14 初次导入时从独立目录重新编译：19 个源文件与冻结提交一致；生成的五页 PDF 与当时的 `paper.pdf` 逐页文字及渲染像素一致。Figure 1 更新后再次编译整稿及独立主图，完成三轮图像检查及双栏实际尺寸检查；无 overfull box、未解析引用或字体替换警告。时间戳、IoU 与配对统计仍取自未改动的 `figures/overview_data.tex`。本轮仅关闭 `ISSUES.md` 中已验证的图示事项。

第二轮风格调整参考了 [Sound-VECaps（ICASSP 2025），Fig. 1](https://personalpages.surrey.ac.uk/w.wang/papers/Yuan%20et%20al_b_ICASSP_2025.pdf)、[Counterfactual Audio（ICASSP 2024），Fig. 2](https://arxiv.org/pdf/2401.04935) 和 [Text-to-Audio Grounding（ICASSP 2021），Fig. 2](https://arxiv.org/pdf/2102.11474) 的流程组织，不复用其图形资产或科学组件。按画图 skill 的层级、连线与论文尺寸检查原则，去掉装饰性大卡片，采用白底、浅色模块、语义化训练虚线框及底部子图标签；9 pt 正文字号不变，画布高度从 7.12 cm 减至 6.22 cm。三轮渲染分别修正标题裁切、输入汇入与行标签间距、损失公式间距，并检查最终整稿和灰度预览。纯矢量 PDF/SVG、PNG 与原生 TikZ 同步导出；数据宏文件哈希与首轮改图一致。整稿分页仍按 `ISSUES.md` 的 F01/F05 待后续统一整理。
