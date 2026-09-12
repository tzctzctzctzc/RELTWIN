# RelTwin timing-cue stress — inference only

Frozen before stress-set prediction inspection, 2026-09-13. This is the final added diagnostic for the paper-polish goal, not a parameter search or independent natural benchmark. The previously considered CompA run remains cancelled.

## Question and single changed factor

The original ESC-50 relation construction couples layout with silence: AB-first uses 1.5 s lead, .25 s within-window gap, 3 s between windows; BA-first uses 2 s, .65 s, 5 s. This permits timing cues correlated with the target layout. Test the fixed models after making that timing pattern identical in both layouts. This simultaneously changes several silence durations, constituting one timing-pattern intervention; it does not isolate each gap's causal effect.

Use every one of the 80 existing development audios, but only the predefined `followed_by` template: 160 queries, 80 inverse pairs. No filtering by model predictions. Extract the two five-second event waveforms from the existing PCM audio. Verify that the duplicated occurrences match exactly. Recompose each original layout with 1.5 s lead, .25 s internal gap, 3 s inter-window gap and 2 s tail. New target windows are [1.5,11.75] and [14.75,25.0]; each recording is 27 seconds. Keep AB-first/BA-first balanced. AB-first audio must be byte-equivalent at decoded-sample level to its original; BA-first changes only the timing pattern. Class labels, sources, excerpts and query text are unchanged.

## Models, metrics, and budget

Evaluate `sft_current_seed0` from the frozen runtime bridge and existing current-environment `no_exchange_seed0` (Cand). Exactly two fixed checkpoints, 160 predictions each, no training, no alternate template or timing search. Freeze code/protocol/audio/manifest/adapter hashes before inference. Wait until the bridge queue has completed, so jobs do not compete on the single GPU. Stop after 3,600 seconds cumulative evaluation wall time, retaining failure evidence.

Primary diagnostic: Cand minus matched SFT JointPairAcc on the 40 changed BA-first audios (80 queries, 40 pairs). Also report mIoU, Swap error, exact same-answer pairs, all 80 audios, unchanged AB-first stratum, and the corresponding original-timing predictions from the bridge/control caches. Group bootstrap uses 20,000 draws by original-source connected component; shared queries remain together. On strata, reuse source groups computed on the full split, rather than claiming that selection breaks source dependence.

This is a post-development stress test motivated by an audited shortcut, not a pristine model-selection gate. The uniform timing pattern is fixed before scores are observed. If both models fail or the candidate advantage disappears, narrow the paper to the original constructed setting and explicitly report that the shortcut challenge is unresolved. If the advantage remains, claim robustness to this specific timing-cue removal only, not all synthetic shortcuts or natural-scene generalization. No follow-up tuning in this goal.
