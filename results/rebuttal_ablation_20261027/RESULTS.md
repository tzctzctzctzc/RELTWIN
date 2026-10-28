# RelTwin 748 ablation results

| Arm | SpotSound mIoU | Relation mIoU | JointPairAcc | Swap pairs | Same answers |
|---|---:|---:|---:|---:|---:|
| s0_candidate | 59.411400 | 87.237467 | 80.333333 | 24 | 1 |
| s0_paired_sft | 58.705857 | 75.992492 | 55.666667 | 110 | 59 |
| s0_candidate_js | 59.126864 | 87.274636 | 80.333333 | 23 | 1 |
| s1_paired_sft | 58.564426 | 79.298093 | 61.666667 | 84 | 47 |
| s1_candidate | 59.326816 | 85.368894 | 78.000000 | 21 | 0 |
| s2_paired_sft | 58.126301 | 74.206161 | 53.333333 | 117 | 66 |
| s2_candidate | 59.428658 | 86.013113 | 78.333333 | 23 | 0 |
| s0_continue128 | 59.480903 | 91.253826 | 91.333333 | 17 | 0 |
| s0_setpo128 | 60.071911 | 92.351781 | 94.666667 | 11 | 0 |
| s0_js_only | 58.735310 | 76.648314 | 56.666667 | 106 | 58 |

All evaluations run on 748. Existing seed-0 checkpoints are reevaluated, not retrained.
The 256+128 LoRA-only comparison is a mechanism extension, not the paper 384+128 full recipe.
All seeds/directions are retained. Cluster bootstrap intervals are not multi-seed training intervals.
