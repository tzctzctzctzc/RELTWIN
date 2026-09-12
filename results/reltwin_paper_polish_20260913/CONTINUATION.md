# RelTwin 投稿打磨续办

2026-09-13。目标模式已开启，交付中英文 Markdown、PDF 和 Overleaf ZIP。当前新分支 `codex/reltwin-paper-polish-20260913`；协议与队列代码冻结提交 `3352742`。旧 50/50 审稿补强实验已完成，不重启。

## 当前小实验

服务器 `connect.westc.seetacloud.com:56527`，工程 `/root/autodl-tmp/SpotSound-ICASSP`。新干净 worktree `spotsound_reltwin_polish_20260913`。Python `env-conda/bin/python`，RTX 4080 32 GiB。

输出 `outputs/reltwin_paper_bridge_20260913`。队列已完成 **6/6**，状态 COMPLETE。训练权重留服务器，全部轻量缓存已取回并提交 `bridge_complete/`。不得重启。

计划六步：SFT seed 0 训练 256 步；关系与公开集各 10 条 smoke；完整关系 320 与 SpotSound 400；关系分析。冻结 752 个代码／权重／数据文件，硬件与软件版本、训练源码和对照数据哈希已与旧队列一致性检查通过。累计任务墙钟上限 7,200 秒；此次没有超参调优。输出权重留服务器，取回预测、summary、freeze、status、budget 和日志。

## 外部诊断判断

核查 CompA 作者项目页与官方 `evaluation/benchmark_eval.py`：官方任务为音频—描述匹配，不是区间定位。公开下载在 Google Drive，本地和服务器均连接超时。没有取得可核验的局部标注或公开数据包，不拼接新数据冒充官方自然定位测试。本轮先取消该外部诊断，保留“自然关系泛化未证实”这一局限；不扩大下载、训练或 benchmark 任务。

## 论文方向

**最后小诊断也已完成 2/2：** `experiments/protocols/RELTWIN_TIMING_STRESS_20260913.md`，预先冻结提交 `efc5ca2`。不是 CompA 外部测试。它统一静音模式，保留全部 80 段声音、原查询和布局。只取 `followed_by` 模板 160 条，两个固定模型共 320 次推理，无训练。输出 `outputs/reltwin_timing_stress_20260913`，轻量缓存已取回 `stress_complete/`。审计 `timing_stress_audit.json` 完成：变化的 40 段 BA-first 上 Cand−SFT JointPairAcc +20.00 点，来源 CI [2.94,33.33]；未变化 40 段的预测全部精确复现。本轮不再增加实验。

以“真实存在但不回答当前查询”的候选窗口为核心，强调同音频两个关系均成立而局部答案不同。主要方法是序列监督加两候选交叉熵（既有 no_exchange），不是 JS 新公式。三种子 JS 增量及 SetPO 对等步数增量均跨零，保留并降低主张。

桥接最终结果：Cand 相较同环境 SFT，关系 mIoU +13.97、JointPairAcc +31.25，SpotSound +1.02；对应来源/音频组 CI 均不跨零。只有一个匹配种子，不与三种子 Cand 对 official 的区间混用。`evidence_complete.json` 和 `FINAL_RESULTS.md` 已记录完整数值。历史其它对照仍有训练环境限制。

**当前剩余仅交付：** 新英文源 `paper/overleaf_icassp2027_reltwin_final_v4/` 已集成两项新结果，正在压缩到四页技术内容并校验。中文 `paper/ICASSP_DRAFT_ZH_v8_BINDING.md` 已更新。接下来导出英文 Markdown、复制证据至包内 notes、数值/编译/逐页视觉检查、Git 提交、生成可导入 Overleaf 的根目录 ZIP 和预览 PDF、同步新分支到远端（不 push GitHub）。交付后更新此记录并删除 heartbeat。

主表文献成绩引用 SpotSound Table 3，不引用 87.2 的 Clotho 消融。完整系统的 Clotho 86.855592 来自 official+边界，不能作为 RelTwin 迁移。UnAV 协议受限，不参加公平排名。原用户 v1/v2/v4 的未提交文件保持不动。

## 最终检查

ICASSP 2027 官方 Paper Kit 当前列出的 regular 截止为 2026-09-16；英文、非双盲、四页技术内容、可第五页引用。按更保守口径把第五页只用于参考文献。所有字至少 9 pt。作者名单已提供；ORCID 和通讯联系人需作者在投递前自行确认。按政策如实保留 AI 辅助披露。

每阶段保存 Git，不自动 push、merge 或投递。现有 heartbeat `reltwin` 已更新到本目标；全部交付完成后删除它。不要在文件或命令行保存 SSH 密码。
