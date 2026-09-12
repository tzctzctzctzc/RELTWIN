# RelTwin review controls — frozen before new training

Date: 2026-09-12. Scope: strengthen the relation-training manuscript using existing evidence and targeted controls, not tune test scores.

## Priority and interpretation

1. Reconstruct official/SFT/RBEE/SetPO relation diagnostics from existing predictions. Add joint paired accuracy (both target IoUs >= .5 AND both correct-window overlaps strictly exceed opposite-window overlaps). Retain the original metrics. Resample complete audio clusters, synchronize sampled clusters across paired models and seeds, and audit shared source excerpts. The old relation split remains development data.
2. Compare RBEE with the exact same two-candidate cross-entropy objective but exchange weight zero. This is not the existing hinge-ranking mode. All other training settings remain fixed: official parent, 256 steps, lr=5e-6, temperature=1, method weight=1, rehearsal weight=.5, cache=512, seeds 0/1/2.
3. Compare each historical SetPO adapter with an additional 64-step continuation of its exact RBEE parent, using the original RBEE objective, fresh AdamW, lr=2e-6, method weight=1, exchange weight=1, rehearsal weight=.5, seed matched. This is an equal-update-count control, NOT an equal-FLOP control. SetPO's six-candidate objective and rehearsal policy differ and are not separately isolated here.

All three seeds run regardless of favorable or unfavorable seed-0 findings. No threshold, learning rate, model selection, or extra module is tuned using these results. Do not suppress failed controls or select the best seed. Primary mechanism endpoint: relation joint paired accuracy, with original PairAcc, Swap error, and mIoU also reported. External endpoint: full 400-query SpotSound mIoU. The diagnostic joint metric was proposed after observing the old results and is explicitly post-hoc.

## Comparable evaluation

Evaluate each new model and each matched historical RBEE/SetPO parent using the same frozen evaluator on this server: complete 320-query RelTwin development and complete 400-query SpotSound. No NOVA routing or boundary editing is applied. Re-evaluate official/SFT seed 0 as bridges. Old cached scores remain historical references; newly generated scores form the new comparison. Ten-query GPU smoke checks precede the queue. Check exact row indices, audio names, queries, annotations, uniqueness, and recorded duration. Preserve individual predictions.

The original pipeline used three-seed final checkpoints, not selected seed maxima. Verify recorded adapter hashes, data hashes, code hashes, dependency versions, and GPU in the run freeze. A copied dirty release repository is not edited; run in a new worktree from the current paper/code branch. The queue uses a process lock, task completion checks, resumable evaluation, and refusal to overwrite partial adapter saves.

## Decision rules

Report effects and paired cluster confidence intervals without requiring a favorable outcome to publish the table. If RBEE does not exceed the no-exchange control on the relation endpoint, do not present exchange consistency as an independently validated main contribution. If SetPO does not exceed equal-step RBEE continuation, describe it as an alternative continuation rather than an independently established objective benefit. A positive development contrast does not establish natural-relation generalization. Novelty against nearby literature and independent natural-relation validation remain separate open requirements.

## Artifacts

Frozen runtime configuration and input/code/checkpoint hashes; per-task logs; per-model training summaries, complete prediction rows and relation analyses; machine-readable queue state; Markdown experiment log; paired audio/source-dependence audit; updated Chinese and English drafts after measured results are available. Baseline files and current result pointers remain unchanged.
