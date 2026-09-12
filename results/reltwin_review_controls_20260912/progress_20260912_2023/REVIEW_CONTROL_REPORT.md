# RelTwin 审稿补强对照

完整队列：False；状态：RUNNING。缺少的计划种子不得用已有正结果替代。

只比较本服务器统一评估流程下的完整 320/400 条预测，不拼接历史总分。来源组区间仅用于关系开发数据。

| 模型 | SpotSound mIoU | 关系 mIoU | PairAcc | Swap error | JointPairAcc |
|---|---:|---:|---:|---:|---:|
| continue_rbee_seed0 | 59.309699 | 89.177327 | 85.625000 | 6.250000 | 85.000000 |
| continue_rbee_seed1 | pending | 88.913321 | 86.875000 | 5.000000 | 86.875000 |
| no_exchange_seed0 | 59.432873 | 88.546199 | 85.000000 | 7.500000 | 84.375000 |
| no_exchange_seed1 | 59.134763 | 86.889929 | 80.000000 | 8.125000 | 80.000000 |
| no_exchange_seed2 | 58.920999 | 85.704206 | 80.000000 | 8.750000 | 80.000000 |
| rbee_seed0 | 59.482210 | 87.928993 | 83.750000 | 8.750000 | 83.125000 |
| rbee_seed1 | 59.289843 | 86.682494 | 81.875000 | 6.875000 | 81.875000 |
| rbee_seed2 | 59.109743 | 85.720020 | 80.000000 | 8.750000 | 80.000000 |
| setpo_seed0 | 59.158605 | 90.246765 | 87.500000 | 6.250000 | 87.500000 |

## exchange_increment

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | True | -0.270 | [-0.642, +0.076] | [-0.700, +0.059] |
| relation / PairAcc@0.5 | True | +0.208 | [-0.625, +1.042] | [-0.877, +0.958] |
| relation / SwapError | True | +0.000 | [-0.625, +0.625] | [-0.424, +0.758] |
| relation / JointPairAcc@0.5 | True | +0.208 | [-0.625, +1.042] | [-0.877, +0.958] |
| public / mIoU | True | +0.131 | [-0.140, +0.439] | n/a |
| public / R1@0.3 | True | -0.250 | [-1.010, +0.498] | n/a |
| public / R1@0.5 | True | +0.250 | [-0.169, +0.756] | n/a |
| public / R1@0.7 | True | +0.250 | [-0.169, +0.754] | n/a |
| public / query_macro_event_F1@0.5 | True | -0.088 | [-0.451, +0.280] | n/a |

## SetPO_minus_equal_updates

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | False | +1.069 | [-0.110, +2.219] | [+0.108, +2.345] |
| relation / PairAcc@0.5 | False | +1.875 | [-1.250, +5.000] | [-0.714, +5.385] |
| relation / SwapError | False | +0.000 | [-2.500, +2.500] | [-2.899, +2.174] |
| relation / JointPairAcc@0.5 | False | +2.500 | [-0.625, +6.250] | [+0.000, +6.250] |
| public / mIoU | False | -0.151 | [-0.677, +0.369] | n/a |
| public / R1@0.3 | False | -0.500 | [-1.980, +0.763] | n/a |
| public / R1@0.5 | False | +0.000 | [-1.000, +1.003] | n/a |
| public / R1@0.7 | False | +0.000 | [-1.000, +1.003] | n/a |
| public / query_macro_event_F1@0.5 | False | -0.671 | [-1.837, +0.354] | n/a |

## RBEE_minus_SFT_seed0

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## RBEE_minus_official

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## SetPO_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | False | +2.318 | [+1.154, +3.654] | [+1.141, +3.658] |
| relation / PairAcc@0.5 | False | +3.750 | [+0.625, +7.500] | [+0.521, +8.219] |
| relation / SwapError | False | -2.500 | [-5.000, -0.625] | [-5.696, -0.515] |
| relation / JointPairAcc@0.5 | False | +4.375 | [+1.250, +8.125] | [+0.847, +8.904] |
| public / mIoU | False | -0.324 | [-0.774, +0.132] | n/a |
| public / R1@0.3 | False | -0.500 | [-1.985, +0.763] | n/a |
| public / R1@0.5 | False | +0.000 | [-1.003, +0.998] | n/a |
| public / R1@0.7 | False | -1.250 | [-2.525, +0.000] | n/a |
| public / query_macro_event_F1@0.5 | False | -0.089 | [-1.236, +1.114] | n/a |

## continuation_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | False | +1.740 | [+0.826, +2.822] | [+0.648, +2.630] |
| relation / PairAcc@0.5 | False | +3.438 | [+1.250, +5.938] | [+0.847, +6.044] |
| relation / SwapError | False | -2.188 | [-4.375, -0.625] | [-3.873, -0.379] |
| relation / JointPairAcc@0.5 | False | +3.438 | [+1.250, +5.938] | [+0.847, +6.044] |
| public / mIoU | False | -0.173 | [-0.777, +0.399] | n/a |
| public / R1@0.3 | False | +0.000 | [-1.250, +1.247] | n/a |
| public / R1@0.5 | False | +0.000 | [-1.003, +1.000] | n/a |
| public / R1@0.7 | False | -1.250 | [-2.750, +0.249] | n/a |
| public / query_macro_event_F1@0.5 | False | +0.582 | [-0.501, +1.808] | n/a |

## 归因状态

- exchange_increment: POSITIVE_MEAN_WITH_UNCERTAIN_CLUSTER_INTERVAL
- SetPO_minus_equal_updates: PENDING_ALL_PLANNED_SEEDS

全部差值为后者减前者；Swap error 越低越好。区间条件于已观察到的种子，不能消除开发选择偏差。原历史模型与新训练对照共享评估流程，不等于训练硬件／软件完全相同；需要结合默认目标复现实验核查该残余混杂。未做自动提交、默认模型替换或参数选择。
