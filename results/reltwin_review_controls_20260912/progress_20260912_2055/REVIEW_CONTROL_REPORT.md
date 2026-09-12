# RelTwin 审稿补强对照

完整队列：False；状态：RUNNING。缺少的计划种子不得用已有正结果替代。

只比较本服务器统一评估流程下的完整 320/400 条预测，不拼接历史总分。来源组区间仅用于关系开发数据。

| 模型 | SpotSound mIoU | 关系 mIoU | PairAcc | Swap error | JointPairAcc |
|---|---:|---:|---:|---:|---:|
| continue_rbee_seed0 | 59.309699 | 89.177327 | 85.625000 | 6.250000 | 85.000000 |
| continue_rbee_seed1 | 59.162459 | 88.913321 | 86.875000 | 5.000000 | 86.875000 |
| continue_rbee_seed2 | pending | 89.042381 | 85.000000 | 6.250000 | 85.000000 |
| no_exchange_seed0 | 59.432873 | 88.546199 | 85.000000 | 7.500000 | 84.375000 |
| no_exchange_seed1 | 59.134763 | 86.889929 | 80.000000 | 8.125000 | 80.000000 |
| no_exchange_seed2 | 58.920999 | 85.704206 | 80.000000 | 8.750000 | 80.000000 |
| rbee_seed0 | 59.482210 | 87.928993 | 83.750000 | 8.750000 | 83.125000 |
| rbee_seed1 | 59.289843 | 86.682494 | 81.875000 | 6.875000 | 81.875000 |
| rbee_seed2 | 59.109743 | 85.720020 | 80.000000 | 8.750000 | 80.000000 |
| setpo_seed0 | 59.158605 | 90.246765 | 87.500000 | 6.250000 | 87.500000 |
| setpo_seed1 | 59.405739 | 89.401225 | 86.250000 | 5.625000 | 86.250000 |

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
| relation / mIoU | False | +0.779 | [-0.204, +1.732] | [-0.089, +2.043] |
| relation / PairAcc@0.5 | False | +0.625 | [-2.500, +3.750] | [-1.744, +4.464] |
| relation / SwapError | False | +0.312 | [-2.188, +3.125] | [-2.778, +2.778] |
| relation / JointPairAcc@0.5 | False | +0.938 | [-2.188, +4.062] | [-1.577, +4.839] |
| public / mIoU | False | +0.046 | [-0.372, +0.437] | n/a |
| public / R1@0.3 | False | -0.250 | [-1.128, +0.515] | n/a |
| public / R1@0.5 | False | +0.125 | [-0.623, +0.877] | n/a |
| public / R1@0.7 | False | +0.125 | [-0.741, +0.886] | n/a |
| public / query_macro_event_F1@0.5 | False | +0.037 | [-0.752, +0.810] | n/a |

## RBEE_minus_SFT_seed0

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## RBEE_minus_official

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|

## SetPO_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | False | +2.518 | [+1.421, +3.766] | [+1.467, +3.936] |
| relation / PairAcc@0.5 | False | +4.062 | [+0.938, +7.812] | [+1.271, +8.456] |
| relation / SwapError | False | -1.875 | [-4.375, +0.312] | [-4.444, +0.000] |
| relation / JointPairAcc@0.5 | False | +4.375 | [+0.938, +8.125] | [+1.587, +8.750] |
| public / mIoU | False | -0.104 | [-0.409, +0.225] | n/a |
| public / R1@0.3 | False | +0.000 | [-0.735, +0.741] | n/a |
| public / R1@0.5 | False | +0.125 | [-0.504, +0.758] | n/a |
| public / R1@0.7 | False | -0.625 | [-1.838, +0.379] | n/a |
| public / query_macro_event_F1@0.5 | False | +0.336 | [-0.438, +1.219] | n/a |

## continuation_minus_RBEE

| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |
|---|---|---:|---|---|
| relation / mIoU | True | +2.267 | [+1.356, +3.303] | [+1.192, +3.110] |
| relation / PairAcc@0.5 | True | +3.958 | [+1.667, +6.875] | [+1.366, +6.386] |
| relation / SwapError | True | -2.292 | [-4.583, -0.625] | [-3.704, -0.625] |
| relation / JointPairAcc@0.5 | True | +3.958 | [+1.667, +6.875] | [+1.366, +6.386] |
| public / mIoU | False | -0.150 | [-0.536, +0.274] | n/a |
| public / R1@0.3 | False | +0.250 | [-0.377, +1.000] | n/a |
| public / R1@0.5 | False | +0.000 | [-0.623, +0.623] | n/a |
| public / R1@0.7 | False | -0.750 | [-1.899, +0.375] | n/a |
| public / query_macro_event_F1@0.5 | False | +0.299 | [-0.520, +1.198] | n/a |

## 归因状态

- exchange_increment: POSITIVE_MEAN_WITH_UNCERTAIN_CLUSTER_INTERVAL
- SetPO_minus_equal_updates: PENDING_ALL_PLANNED_SEEDS

全部差值为后者减前者；Swap error 越低越好。区间条件于已观察到的种子，不能消除开发选择偏差。原历史模型与新训练对照共享评估流程，不等于训练硬件／软件完全相同；需要结合默认目标复现实验核查该残余混杂。未做自动提交、默认模型替换或参数选择。
