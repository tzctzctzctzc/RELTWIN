# RelTwin review controls — experiment log

State: COMPLETE. Completed tasks: 50/50.

No new hyperparameter selection; unfavorable results are retained.

| Model | Steps | Exchange | Relation mIoU | PairAcc | Swap error | SpotSound mIoU |
|---|---:|---:|---:|---:|---:|---:|
| continue_rbee_seed0 | 64 | 1.0 | 89.177327 | 85.625000 | 6.250000 | 59.30969933911936 |
| continue_rbee_seed1 | 64 | 1.0 | 88.913321 | 86.875000 | 5.000000 | 59.16245857870338 |
| continue_rbee_seed2 | 64 | 1.0 | 89.042381 | 85.000000 | 6.250000 | 59.05835209281087 |
| no_exchange_seed0 | 256 | 0.0 | 88.546199 | 85.000000 | 7.500000 | 59.43287334410579 |
| no_exchange_seed1 | 256 | 0.0 | 86.889929 | 80.000000 | 8.125000 | 59.13476292499562 |
| no_exchange_seed2 | 256 | 0.0 | 85.704206 | 80.000000 | 8.750000 | 58.9209990681924 |
| official | existing | existing | 37.644558 | 0.625000 | 65.000000 | 58.41535280888225 |
| rbee_seed0 | existing | existing | 87.928993 | 83.750000 | 8.750000 | 59.48220985389601 |
| rbee_seed1 | existing | existing | 86.682494 | 81.875000 | 6.875000 | 59.289843354043306 |
| rbee_seed2 | existing | existing | 85.720020 | 80.000000 | 8.750000 | 59.109743098339706 |
| review_summary | existing | existing | pending | pending | pending | pending |
| setpo_seed0 | existing | existing | 90.246765 | 87.500000 | 6.250000 | 59.158605231319925 |
| setpo_seed1 | existing | existing | 89.401225 | 86.250000 | 5.625000 | 59.40573880209008 |
| setpo_seed2 | existing | existing | 89.792560 | 86.250000 | 5.625000 | 59.023123630762285 |
| sft_seed0 | existing | existing | 75.337160 | 55.625000 | 38.125000 | 58.756570253219124 |
