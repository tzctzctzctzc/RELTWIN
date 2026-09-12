# RelTwin — ICASSP 2027 审稿补强 P0 论文包

## 使用

在 Overleaf 中选择“新建项目 → 上传项目”，上传交付的 ZIP。入口为 `main.tex`，编译器为 **pdfLaTeX**。无需 shell escape、外部模型、私有数据或额外图片下载；图 1 使用源码内 TikZ 矢量绘制。

已填入 Zhicheng Tang、Yuehan Zhang，两位作者均为 Huazhong University of Science and Technology，邮箱沿用用户提供的信息。通讯作者尚未指定，`authors.tex` 中保留注释 TODO，不输出虚构身份。

## 与原版的变化

主线维持同音频反向关系配对学习。新增 SFT 关系诊断，使公开集与关系开发集的对照对应同一个训练阶段；新增 JointPairAcc，排除仅覆盖两个窗口即可通过 0.5 阈值的解释；按音频及原始录音来源重新计算关系收益区间。标题不再提前将效果归给尚未隔离的交换项。旧版论文和结果均保留。

去交换项和等更新步数继续训练的三种子对照已冻结并在服务器后台运行。**本 ZIP 只包含已完成的缓存审计，不包含尚未产出的训练对照结果；这是 P0 审阅稿，不是最终投稿稿。**

论文包含中英文对应的摘要、引言、方法定义、实验设置、主表、训练阶段分析、关系诊断、讨论与结论。中文完整稿在 `notes/source_zh.md`，它不参与 LaTeX 编译。

三张表分别承担不同功能：表 1 与 SpotSound 原文主表成绩比较；表 2 定位公开集训练阶段收益；表 3 同时展示 SFT/RBEE 的 seed 0 对照和 RBEE/SetPO 三种子阶段变化。图 1 是方法示意图，不是新实验或真实失败案例。

Clotho、UnAV 的已有完整系统结果使用 official 初始预测，因此表 1 明确列为分数据集配置，不将其成绩包装成 RelTwin 模型的迁移收益。动态半径不再作为核心创新。

## 已完成的本地检查

MiKTeX pdfLaTeX + BibTeX 编译检查及摘要词数见 `notes/build_validation.json`。模板 `spconf.sty` 和 `IEEEbib.bst` 与原用户模板逐字节一致；US Letter、10 pt 正文、9 pt 表格，无页码。目标版式为 **4 页技术内容 + 1 页参考文献**。

主表 11 个系统行和差值行逐项核对中文稿；训练阶段均值、标准差、关系指标核对已有 JSON。PDF 全部字体嵌入，无 Type 3 字体、未解析引用或水平越界，已逐页检查五页渲染。

`notes/build_validation.json` 记录页数、字体、边界和 PDF 哈希；`notes/evidence_summary.json` 记录原始实验文件／方法代码哈希及核心数值。`notes/relation_audit.json` 和 `notes/RELATION_AUDIT.md` 为本次已完成的事后统计审计，包含完整分组置信区间。若修改正文，请重新编译并核查页数。

## 文件结构

- `main.tex`、`authors.tex`：入口和作者。
- `sections/`：英文正文。
- `figures/relation_pair.tex`：可编辑矢量示意图。
- `tables/`：文献主表、训练阶段表、关系诊断表。
- `references.bib`、`IEEEbib.bst`：参考文献。
- `spconf.sty`：用户提供的 ICASSP 模板样式，未修改。
- `notes/`：中文版、证据和排版核对、投稿前清单。

本地编译顺序：

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## 使用边界

这是依据已有证据重写的可编译研究草稿，不是已经完成创新性认证或作者审核的最终投稿。全文保留阶段归因、统计口径和协议说明；不能通过删除这些说明，将未知结论变成已验证结果。`notes/SUBMISSION_CHECKLIST.md` 列出仍需作者处理的事项。

官方格式规则已于 2026-09-12 核对：[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)、[Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)。本稿采取第五页只放参考文献的保守安排，AI 文稿辅助披露位于前四页。实际投稿系统的最终检查仍需作者完成。
