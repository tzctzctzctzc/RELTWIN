# RelTwin v36 有限范围增量修改

## 2026-09-23：基于合作者融合版 83d92d3 的页末小修

按作者要求，删除致谢中的 GPU 资助、无机构/企业经费及无利益冲突三句；保留原有公开数据和伦理说明。结论末尾增加一句，强调从声音类别识别到查询指定的实例选择。仅修改这两处正文与维护记录，更新固定 `paper.pdf`；未改表格、实验或图。编译并检查第 4、5 页，维持四页正文与第五页声明/参考文献，未改变字体、模板或页边距。以下为此前 v36 修改记录。

日期：2026-09-23。底稿为作者确认的 GitHub `paper` 提交 `b28b3e9`；开始时工作区干净、与远端一致。没有用微信中的 `paper(1).pdf`、`paper(4).pdf` 或 Teacher Revised PDF 覆盖源码。

## 文件与输出

入口 `main.tex`；分章 `sections/`；文献 `references.bib`；可编辑主图 `figures/overview_art.tex`，数值来自 `figures/overview_data.tex`；单图入口 `overview_figure.tex`。

按作者后续确认，新编译稿统一保存为 **`paper.pdf`**，不另设版本文件名；移除重复命名的 PDF，旧稿可从 Git 历史恢复。本次文件名统一不改变 v36 论文内容。单图 PDF/PNG/SVG 与兼容预览同步更新，均由原图源生成。

## 按章节的修改

| 位置 | 修改 | 理由 |
|---|---|---|
| Abstract | 限定 31.25 点为 SpotSound-A、Stage 1、160 对逆查询上的 matched candidate-supervision 对照；保留公开基准与 mIoU 数字 | 使证据身份明确 |
| Abstract / Introduction / Method | 将指输出集合基数的 event count 改为 interval count | 与 n_S 定义一致，不机械替换其他 event |
| Introduction / related_work.tex | 相关工作拆为定位生成、组合学习与负样本两段；保留引用及技术描述 | 改善阅读节奏，不增加章节 |
| Introduction | 明确两个窗口真实存在，但同一查询只有一个匹配；精简末段重复指标和评测说明 | 解释 locally valid，给摘要准确性补充留出篇幅 |
| Method 2.2 | replay 以保留普通定位监督的目的引入；候选对比、JS 对齐定义保留 | 现有正确说明不重复重写 |
| Method 2.3 | 用分级偏好解释为何区分不完美答案；解释冻结分布和 KL 的设计目的 | 补足动机，不增添独立消融结论 |
| Figure 1 | `Compare answers + exchange labels` → `Competition + reversal consistency` | 与 JS 定义对应，保留字号、坐标、六候选、时间条、IoU |
| authors.tex | 五位姓名分别增加 ORCID 超链接，邮箱、顺序和贡献标记不变 | 使用现有 hyperref，不挤占首页 |
| Acknowledgements / References | 检查后保持原文和元数据不变 | 没有需要修改的事实或可靠的新来源 |

一级目录保持六章。EXPERIMENTS、CONCLUSION 和表格未改；建议集中于 `HANDOFF_NOTES.md`，其中包含可直接替换的英文结论及消融说明。

## 检查与边界

使用原有 pdfLaTeX + BibTeX 流程编译；保留模板、字号、页边距。首次编译超页后，仅压缩 Introduction 重复文字，恢复四页正文与第五页声明、文献。

稿内检查：19 个实际引用键与 19 个条目对应，无缺失键、未引用条目或重复键；无未解析交叉引用、`??`、越界公式、缺失字体或编译错误。五个显示公式与底稿完全一致；实验、结果、结论、表格、案例宏、参考文献、模板和版面参数逐文件检查不变。五个 ORCID PDF 链接检查与作者名字对应。

视觉检查：最终 PDF 五页逐页渲染，检查作者区、实际双栏尺寸主图、公式、两表、致谢和文献。没有文字遮挡、裁切或不合理大块留白；主图原有矩阵关系和预测坐标未改。

外部出版信息：本轮未重新访问出版网站，未修改任何 .bib 元数据；不能将稿内键对应检查视作出版真实性核验。历史外部核验记录仍见 ISSUES.md。

未完成事项：双通讯最终作者确认、ORCID 身份/门户绑定、表 1 完整实验归档、表 2 的 1.02 增益原始精度核查，以及合作者对实验和结论建议的确认。无新增实验，不声明投稿终稿。
