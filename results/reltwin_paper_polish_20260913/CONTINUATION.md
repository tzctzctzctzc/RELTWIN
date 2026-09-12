# RelTwin 投稿打磨续办

2026-09-13。目标模式已开启，交付中英文 Markdown、PDF 和 Overleaf ZIP。当前新分支 `codex/reltwin-paper-polish-20260913`；协议与队列代码冻结提交 `3352742`。旧 50/50 审稿补强实验已完成，不重启。

## 当前小实验

服务器 `connect.westc.seetacloud.com:56527`，工程 `/root/autodl-tmp/SpotSound-ICASSP`。新干净 worktree `spotsound_reltwin_polish_20260913`。Python `env-conda/bin/python`，RTX 4080 32 GiB。

输出 `outputs/reltwin_paper_bridge_20260913`。队列 `run_reltwin_paper_bridge.py` 于 2026-09-13 00:31 左右后台启动；已确认 `status.json` 为 RUNNING/train，PID 2929，训练子进程 PID 3072。先核对命令和状态，勿仅用 PID 判断。初始 nohup SSH 调用超时只是父 shell 未退出；独立训练已经运行，不重复启动。

计划六步：SFT seed 0 训练 256 步；关系与公开集各 10 条 smoke；完整关系 320 与 SpotSound 400；关系分析。冻结 752 个代码／权重／数据文件，硬件与软件版本、训练源码和对照数据哈希已与旧队列一致性检查通过。累计任务墙钟上限 7,200 秒；此次没有超参调优。输出权重留服务器，取回预测、summary、freeze、status、budget 和日志。

## 外部诊断判断

核查 CompA 作者项目页与官方 `evaluation/benchmark_eval.py`：官方任务为音频—描述匹配，不是区间定位。公开下载在 Google Drive，本地和服务器均连接超时。没有取得可核验的局部标注或公开数据包，不拼接新数据冒充官方自然定位测试。本轮先取消该外部诊断，保留“自然关系泛化未证实”这一局限；不扩大下载、训练或 benchmark 任务。

## 论文方向

以“真实存在但不回答当前查询”的候选窗口为核心，强调同音频两个关系均成立而局部答案不同。主要方法是序列监督加两候选交叉熵（既有 no_exchange），不是 JS 新公式。三种子 JS 增量及 SetPO 对等步数增量均跨零，保留并降低主张。

先完成文稿概念部分、近邻原文比较和旧／新推理漂移审计。待桥接结束，以同环境 SFT 对 no_exchange seed 0 为主要训练归因对照；若结果削弱主张，按结果收紧，不调参。历史模型的其它对照仍有训练环境混杂，不隐瞒。

主表文献成绩引用 SpotSound Table 3，不引用 87.2 的 Clotho 消融。完整系统的 Clotho 86.855592 来自 official+边界，不能作为 RelTwin 迁移。UnAV 协议受限，不参加公平排名。原用户 v1/v2/v4 的未提交文件保持不动。

## 最终检查

ICASSP 2027 官方 Paper Kit 当前列出的 regular 截止为 2026-09-16；英文、非双盲、四页技术内容、可第五页引用。按更保守口径把第五页只用于参考文献。所有字至少 9 pt。作者名单已提供；ORCID 和通讯联系人需作者在投递前自行确认。按政策如实保留 AI 辅助披露。

每阶段保存 Git，不自动 push、merge 或投递。现有 heartbeat `reltwin` 已更新到本目标；全部交付完成后删除它。不要在文件或命令行保存 SSH 密码。
