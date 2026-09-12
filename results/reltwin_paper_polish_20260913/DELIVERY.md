# ICASSP 2027 投稿打磨交付

交付日期：2026-09-13。本轮受限实验与论文打磨已完成，交由作者审阅，不自动投稿。

## 成品

- [中文研究稿](../../paper/ICASSP_DRAFT_ZH_v8_BINDING.md)：完整问题、方法、主表、受控对照、压力测试和分析；是作者阅读版，内容比四页英文正文展开更多。
- [英文 Markdown](../../paper/ICASSP_DRAFT_EN_v8_BINDING.md)：从最终英文 LaTeX 导出，保留数学公式与相同表格。
- PDF：`I:/AI科研/SpotSound/RelTwin_ICASSP2027_Binding_Preview_20260913.pdf`，255,350 字节。
- Overleaf ZIP：`I:/AI科研/SpotSound/RelTwin_ICASSP2027_Binding_Overleaf_20260913.zip`，67,447 字节；直接上传，以根目录 `main.tex` 为主文件，选择 pdfLaTeX。包含中英文稿、证据 JSON、协议及作者备忘，不含大权重。
- [实验结果说明](FINAL_RESULTS.md)、[机器验收记录](delivery_validation.json)。

源文件：`paper/overleaf_icassp2027_reltwin_final_v4/`。最终论文内容提交 `f70c0ba51c4700059ff340e2e1def42f54d4418b`；之后的提交仅补交付审计与状态记录，不改变归档论文内容。分支 `codex/reltwin-paper-polish-20260913`。

## 完成的核验

已从原始预测重新核验 15 个模型的指标与行对齐，核对完整 SFT 桥接和 160 查询的冻结压力测试。公开文献成绩与 SpotSound 主表逐项对应，不把消融配置当成主表榜首。中文、英文、LaTeX 和包内证据保持一致。数学 Markdown 的行分隔与数字舍入也有测试。

本轮相关测试 **29/29 通过**。pdfLaTeX + BibTeX 完整编译通过：138 词摘要，4 页技术内容，独立第 5 页仅参考文献；US Letter、嵌入字体、无 Type 3、无溢出和未解析引用。沿用用户提供的原模板，`spconf.sty` 与 `IEEEbib.bst` 字节一致；表格和示意图正常文字使用至少 9 pt，未通过缩小字体挤页。

已查看全部 5 页渲染图，检查标题、作者、公式、示意图、三个表格、跨栏边界及页尾，无可见遮挡或裁切。ZIP 在新的 `reltwin_final_v4_clean_build/` 目录从零编译，全部页面文字和渲染像素与交付预览一致，未依赖旧辅助文件。

## 归档哈希

| 文件 | SHA256 |
|---|---|
| Overleaf ZIP | `a09a650280eb6786efbea65476395f60c8d07e6ac16e2e1570c77e89a8697ae2` |
| 预览 PDF | `36a0a7611d376f2fe5c834ebf8fae4fe5f23d37293a707a0bb89e19496f1637b` |
| 完整证据 JSON | `1717d22f3b2cdfa75e971279ef82a4f45f92785697271702cc8e087ae567a5c2` |
| 压力测试审计 JSON | `3a2e840782b4f72addac58ea96bafb4c7de3846525576c3daeb585be14d1e4b6` |

## 作者下一步

核读科学主张与相邻工作区别，确认作者顺序、邮箱、ORCID、投稿联系人、数据/代码使用及 AI 辅助披露，然后在投稿系统完成文档检查。未提供的 ORCID 或通讯身份未编造。独立自然关系验证、单个匹配 SFT 种子、开发数据已被观察和等步数非等算力仍是本文边界。

本轮将主要实证依据从微小边界涨分转到同数据候选训练与查询绑定诊断，补实验任务累计约 36.57 分钟。改善稿件的证据与表达不等于保证录用，不能据此给出可靠的中稿百分比。没有改默认模型结果指针，没有覆盖用户原有未提交稿件。
