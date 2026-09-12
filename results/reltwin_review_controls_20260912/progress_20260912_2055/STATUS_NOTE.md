# 2026-09-12 20:55 续办检查

队列正常，40/50 个步骤完成。SetPO 与等更新步数 RBEE continuation 的 seed 0、1 已完成关系和 SpotSound 的完整配对评估。seed 2 continuation 已训练完成，正在运行 400 条公开集评估；其后的 SetPO seed 2 与 official/SFT 基线重评仍按原队列进行。

因新增完整 seed 1 对照，已更新严格审计并保存本快照。两种子平均 SetPO − equal-updates：关系 mIoU 约 +0.779 点、JointPairAcc +0.9375 点、SpotSound mIoU 约 +0.046 点，相关分组区间跨零。该结果不完整，主结论仍为 `PENDING_ALL_PLANNED_SEEDS`；不能据此评价三种子最终效果，也不据此调整策略。

去交换项三种子结论不变。训练源码、冻结协议、基线、默认预测与 P0 文稿均未改动，自动续办继续。全队列完成后再汇总统计、核查重评漂移并更新下一版论文。
