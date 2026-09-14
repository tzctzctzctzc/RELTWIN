# Frozen RelTwin cross-benchmark completion

The author requested immediate completion of transfer evaluations for the current
RelTwin method, not reuse of archived NOVA-system scores. No training is performed.

## Fixed method and tasks

- Use the existing `no_exchange_seed0` adapter (256 updates, candidate CE plus
  sequence supervision and category rehearsal; exchange coefficient zero).
- Evaluate all seven remaining releases regardless of scores: UnAV 100, TUT 104,
  AudioGrounding-v2 997, DESED 1112, Clotho 6649, AEGBench public-v3 9924, LAT 426.
- Then evaluate the existing matched SFT seed-0 adapter on UnAV, AudioGrounding,
  and Clotho. This is three additional inference runs, not additional training.
- Fixed seed 0 is the paper's matched control identity, not a seed chosen using
  the new benchmark results. Do not mix these results with the three-seed mean.
- No score-based gate, threshold changes, best-seed selection, NOVA routing,
  boundary editing, or silent benchmark exclusion.

## Inference and provenance

Reuse the exact paper-era evaluator, processor, and parser after checking their
hashes against the execution freeze. Greedy, bfloat16, 16 kHz mono, 128 output
tokens; AEGBench retains its existing JSON prompt and 256-token protocol.
The plan hashes adapters, base weights/configs, scripts, manifests and audio.
Before full inference, run the UnAV ten-query GPU smoke check. Import only caches
whose adapter identity, query, index, audio, duration, targets, and metrics agree.
Every output directory is new; old predictions and dirty worktrees stay intact.

LAT cannot be silently truncated at 600 seconds. Use the existing label-free
600-second window / 300-second stride existence-ranking wrapper with the selected
adapter, and report the wrapper explicitly, separately from short-audio direct
generation. Keep all 426 released rows, including the known invalid annotations;
the scorer also preserves released-endpoint metrics.

## Prior observation discovered on the new server

An untracked, completed first-1000 Clotho run dated September 14 exists under
`spotsound_reltwin_polish_20260913/results/reltwin_real_bench_20260914`.
It uses the current RelTwin and matched SFT seed-0 adapters. Recorded mIoUs are
89.644274 and 89.743765; its old score gate stopped before full evaluation.
Preserve that result and disclose prior inspection. If strict row validation
passes, import these 1000 rows into a new output and finish all 6649 rows;
otherwise do not reuse or overwrite the old files. That old gate does not apply.

## Metrics and interpretation

Report set-mIoU, R1@.3/.5/.7, event-F1 at IoU .5 and audio-group bootstrap CIs.
Primary short-audio scoring clips intervals to decoded audio duration, matching
the previous release audit; also report the released-endpoint result.
AEGBench additionally reports mean best-match IoU, its different metric family.
Its 9924-query expansion is not interchangeable with the historical leaderboard.
AudioGrounding/UnAV retain the documented source-cardinality caveat in SpotSound
Table 2. Paired SFT differences and comparisons with published methods are separate.

The queue resumes only identity-validated rows, rejects duplicate IDs and damaged
JSONL tails, and records failures without stopping unrelated benchmarks. Markdown,
machine-readable summaries, hashes and predictions are preserved for both positive
and negative outcomes. A background server process survives local disconnection.
