# 审稿补强：执行状态与续办说明

2026-09-12。本轮依用户要求在新服务器继续完善论文；不修改基线、默认结果指针或旧脏仓库。本文不是训练完成报告，实时状态以远端 `status.json` 为准。

## 已完成

新分支 `codex/reltwin-review-controls-20260912`。训练队列代码冻结提交 `c9d27d7`；缓存诊断审计提交 `71ac768`。初始 19 项训练目标／指标测试通过，GPU 的关系 10 条和 SpotSound 10 条 smoke 均通过。另 19 项分组审计与汇总校验测试通过。后台队列共 50 个步骤，包含 6 次新训练和历史 checkpoint 的统一重评。

P0 文稿：`paper/ICASSP_DRAFT_ZH_v7_REVIEW.md`；英文源：`paper/overleaf_icassp2027_reltwin_review_v3/`。123 词摘要，已编译为四页正文加一页参考文献。补齐同 seed SFT 关系诊断、JointPairAcc 和两种依赖分组 bootstrap。两项训练归因对照未完成前，文稿明确保持 pending，不把联合目标收益归给交换项。

## 当前服务器与环境

- SSH：`root@connect.westc.seetacloud.com:56527`。使用用户在当前任务最新提供的凭据；不要写入 Git、日志、命令行参数或定时任务提示。
- SSH 主机公钥 SHA256（十六进制）：`10fb2d83c8cd37ebb706cd49c903aa9417aadf0f9ae15ca7a0901ee3c627215d`。
- 工程根：`/root/autodl-tmp/SpotSound-ICASSP`。
- 新 worktree：`/root/autodl-tmp/SpotSound-ICASSP/spotsound_reltwin_review_20260912`。
- 运行 Python：`/root/autodl-tmp/SpotSound-ICASSP/env-conda/bin/python`。
- 输出根：`/root/autodl-tmp/SpotSound-ICASSP/outputs/reltwin_review_controls_20260912`。
- 队列最初 PID：2606；先核对进程命令和当前状态，不能单靠 PID 判断。nohup 已脱离 SSH。
- `/root/autodl-tmp/SpotSound-ICASSP/spotsound_release` 是旧脏仓库，不操作。
- 只使用当前这台机器，不使用旧端口。新服务器公开 GPU 为 RTX 4080，32 GiB。

队列期间不修改冻结的训练、评估或协议文件。追加汇总脚本和论文可以快进同步，但不能改动 `freeze.json` 中的执行源码来强行续跑。中断时先检查 `status.json`、当前日志和 adapter 完整性；同冻结配置续跑支持完整任务跳过和预测追加，半保存的 adapter 必须先审计。

## 结果审计命令（远端 worktree）

```sh
/root/autodl-tmp/SpotSound-ICASSP/env-conda/bin/python -B scripts/summarize_reltwin_review_controls.py \
  --repository . \
  --runs-dir /root/autodl-tmp/SpotSound-ICASSP/outputs/reltwin_review_controls_20260912 \
  --relation-manifest /root/autodl-tmp/SpotSound-ICASSP/autoresearch/06_experiments/data/reltwin_esc50_v1/test.json \
  --public-manifest /root/autodl-tmp/SpotSound-ICASSP/datasets/SpotSound-Bench/annotations_processed.json \
  --output /root/autodl-tmp/SpotSound-ICASSP/outputs/reltwin_review_controls_20260912/review_summary
```

脚本检查完整行、索引、音频、查询、标注、跨模型时长、原评估 IoU、汇总指标及冻结哈希。未完成的种子明确标记 pending。输出机器报告、Markdown、逐条退步案例。所有比较均为本服务器同评估流程下的 target − source，Swap error 越低越好；CI 条件于已观察种子。历史训练模型在此重评不等于训练软件／硬件完全相同，若差异影响结论，需再做同环境默认 RBEE 复现桥接，不能隐藏此残余混杂。

## 后续优先级

1. 完成三种子去交换项对照，保留两候选交叉熵，只去掉 JS；不按 seed 0 正负停止。主要关系指标为 JointPairAcc，并报告 mIoU、PairAcc、Swap 及全量 SpotSound。
2. 完成三种子 RBEE 继续训练 64 步，与对应 SetPO 的 64 步比较；等更新步数不是等 FLOP，也不分别隔离六候选、软质量标签和回放政策。
3. 新旧 official/SFT/RBEE/SetPO 结果逐行核对，不直接将新训练分数与旧总分相减。统一评估漂移、训练环境残余混杂单列。
4. 结果完整后下载小体积预测、汇总、freeze、日志与分析，保存 Git；新模型权重留在服务器，不纳入 Git。
5. 依据实际结果更新下一版中英文稿。若 JS 无独立增量，主线停留在配对候选学习，不宣称交换正则独立有效；若 SetPO 不优于等步数继续训练，降为可选方案。不能用事后解释回避负对照。
6. 编译、逐页检查、数值校验并交付新 Markdown、PDF、Overleaf ZIP。保留 P0 版本。不得自动推送、合并 main 或投递论文。

独立自然关系泛化和近邻文献核查是后续不同证据需求；当前对照通过也不自动证明这两项。新的实验设置须单独冻结，不在已查看的结果上反复调参。
