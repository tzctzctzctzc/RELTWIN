# NOVA — ICASSP 2027 Overleaf source

## 导入 Overleaf

新建项目 → 上传项目 → 选择交付的 ZIP。主文档选择 `main.tex`，编译器选择 **pdfLaTeX**。正常 Recompile 即可，Overleaf 会自动处理 BibTeX；不需要 shell escape、外部字体、联网下载数据或模型。

本项目为英文会议论文草稿，含摘要、引言、方法公式、实验设置、跨双栏主表、迁移／开发结果小表、结果讨论、结论与参考文献。依据项目 Markdown 和当前实现整理，没有新增实验。`notes/source_zh.md` 保留中文原稿，不参与编译。

## 文件

- `main.tex`：入口、标题、宏、章节顺序及文稿辅助工具披露。
- `authors.tex`：作者顺序、单位、邮箱。已填 Zhicheng Tang、Yuehan Zhang 和华中科技大学；通讯作者尚未指定，TODO 保留在源码注释中，不输出虚构姓名。
- `sections/`：摘要、引言、方法、实验、讨论、结论。
- `tables/`：主表及迁移／开发结果表。
- `references.bib`：已核对的真实文献，采用明确版本的 arXiv 条目，避免编造卷期页码。
- `spconf.sty`、`IEEEbib.bst`：用户提供的模板 ZIP 中原装文件，未修改。
- `notes/SUBMISSION_CHECKLIST.md`：投稿前需要作者处理的事项；不是论文正文。

## 模板与规则

实际找到并使用的压缩包是 `ICASSP2027_Paper_Templates.zip`，不是一个 `_Paper/_Templates.zip` 子目录。压缩包中的 `Template.tex` 首行注释仍写 2026，但正文标题为 ICASSP 2027；排版遵循其原装 `spconf.sty`，未改用 IEEEtran。

采用 US Letter、10 pt 正文、9 pt 表格、双栏、无页码。技术内容控制在前 4 页内，参考文献另起一页；不缩小页面边距或通过整体缩放表格绕过 9 pt 要求。作者名单非匿名。

官方规则核对时间：2026-09-12。

- [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)：排版、作者、摘要约 100–150 词、最多 4 页技术内容和可选第 5 页。
- [Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)：AI 辅助内容披露要求。
- [Submission Instructions](https://cmsworkshops.com/ICASSP2027/papers.php)：投稿规则与作者审核责任。

官网不同页面对第 5 页可包含内容的表述略有差异；本稿采取更保守做法，第 5 页只放参考文献，辅助工具披露置于前 4 页。

## 本地编译

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

或者使用 `latexmk -pdf main.tex`。请勿将编译后的 PDF 自动视为已通过科研内容审查或投稿系统合规审核。

## 本地验证记录

2026-09-12 使用 MiKTeX pdfLaTeX 和 BibTeX 完成编译：5 页，前 4 页为正文，第 5 页只有参考文献；摘要 134 词。主表 10 个对照方法及我们的方法共 11 行逐项核对，与中文原稿一致。没有未解析引用或水平越界；PDF 字体全部嵌入为 Type 1。最终技术页平衡存在约 1.72 pt 的内部 vertical-box 提示，已逐页视觉检查，未见内容裁切或覆盖；这不代表投稿系统验证已完成。数值／版面检查见 `notes/build_validation.json`。修改内容或作者信息后需重新检查排版。

## 结果解释

主表比较完整系统与原论文的主表成绩，不把所有差值归因于边界模块。Clotho 的 mIoU 更高，但两项召回略低；UnAV 有协议限制，AudioGrounding／AEGBench 是开发证据。DESED／TUT 的不利结果保留。本文不声称所有数据集、所有指标均领先，也不声称 bootstrap 提供理论安全保证。

与中文 Markdown 相比，英文稿压缩了重复解释，补入直接对应 `boundary_utility.py` 的方法定义和公式。它不是新的算法实现，不包含新消融结果。
