# RelTwin 审稿补强对照

完整队列：False；状态：RUNNING。缺少的计划种子不得用已有正结果替代。

只比较本服务器统一评估流程下的完整 320/400 条预测，不拼接历史总分。来源组区间仅用于关系开发数据。

| 模型 | SpotSound mIoU | 关系 mIoU | PairAcc | Swap error | JointPairAcc |
|---|---:|---:|---:|---:|---:|
| no_exchange_seed0 | 59.432873 | 88.546199 | 85.000000 | 7.500000 | 84.375000 |
| rbee_seed0 | 59.482210 | 87.928993 | 83.750000 | 8.750000 | 83.125000 |

## exchange_increment

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | False | -0.617 | [-1.430, +0.116] | [-1.335, +0.135] |
| relation / PairAcc@0.5 | False | -1.250 | [-3.125, +0.000] | [-3.571, +0.000] |
| relation / SwapError | False | +1.250 | [+0.000, +3.125] | [+0.000, +3.571] |
| relation / JointPairAcc@0.5 | False | -1.250 | [-3.125, +0.000] | [-3.571, +0.000] |
| public / mIoU | False | +0.049 | [-0.490, +0.623] | n/a |
| public / R1@0.3 | False | -1.000 | [-2.488, +0.252] | n/a |
| public / R1@0.5 | False | +0.750 | [-0.251, +1.980] | n/a |
| public / R1@0.7 | False | +0.500 | [+0.000, +1.250] | n/a |
| public / query_macro_event_F1@0.5 | False | -0.153 | [-1.028, +0.708] | n/a |

## SetPO_minus_equal_updates

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## RBEE_minus_SFT_seed0

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## RBEE_minus_official

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## SetPO_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## continuation_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## 归因状态

- exchange_increment: PENDING_ALL_PLANNED_SEEDS
- SetPO_minus_equal_updates: PENDING_ALL_PLANNED_SEEDS

全部差值为后者减前者；Swap error 越低越好。区间条件于已观察到的种子，不能消除开发选择偏差。原历史模型与新训练对照共享评估流程，不等于训练硬件／软件完全相同；需要结合默认目标复现实验核查该残余混杂。未做自动提交、默认模型替换或参数选择。
