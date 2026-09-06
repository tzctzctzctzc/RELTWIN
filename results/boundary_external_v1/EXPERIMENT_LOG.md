# Frozen Boundary Utility external-benchmark evaluation

Run date: 2026-09-06. Branch: `codex/benchmark-completion-20260906`.

This experiment applies the frozen Boundary Utility model released on
`main@7b3148b` to cached official SpotSound-A predictions. It does not retrain
or tune the utility model, verifier, feature backbone, radius (`0.25` seconds),
or decision margin (`0.02`) on either evaluated test set.

## Results

| Dataset and metric | Incumbent | Selected | Delta | Group-bootstrap 95% CI | Catastrophic regressions | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AudioGrounding-v2, raw-endpoint set-IoU | 70.3901 | 70.3253 | -0.0648 | [-0.3568, 0.2107] | 1 | FAIL |
| AudioGrounding-v2, SpotSound clipped set-IoU | 70.4181 | 70.3534 | -0.0647 | not recomputed | 1 | FAIL |
| AEGBench local expansion, raw-endpoint set-IoU | 44.8032 | 45.5864 | +0.7832 | [0.6666, 0.9047] | 1 | FAIL |
| AEGBench local expansion, Auto-AEG core mIoU | 40.8202 | 41.5526 | +0.7324 | not recomputed | 1 | FAIL |

AudioGrounding switched 50 of 997 rows: 30 wins, 947 ties, and 20 losses.
AEGBench switched 787 of 9,924 rows: 600 wins, 9,186 ties, and 138
losses. The AEGBench gain is positive under every reported Auto-AEG metric,
but both datasets fail the frozen zero-catastrophic-regression safety gate.
Neither result is promoted to the default method.

The raw AudioGrounding audit preserves 132 annotation endpoints that exceed the
materialized waveform by at most 0.0735 seconds. Clipping those endpoints exactly
reproduces the pre-existing 70.4181 incumbent, and does not change the negative
transfer conclusion.

The AEGBench result is explicitly diagnostic. The local manifest expands every
positive category in each downloaded item and does not match the item/query
counts used by the later Auto-AEG v4 paper; it must not be presented as a
directly comparable leaderboard score.

## Failure audit

The single AEGBench catastrophic regression is a five-occurrence `pop` query.
The incumbent tightly tracks 0.04--0.12 second events, while the fixed absolute
trust region widens every occurrence by roughly 0.2 seconds per boundary,
reducing IoU from 0.7295 to 0.2154. AudioGrounding shows the same pattern: losing
switches have a median incumbent interval length of about 0.35 seconds. This
supports a development-only follow-up using duration-relative boundary radii or
a short-event abstention feature; the public test labels were not used to tune
such a rule in this run.

## Reproducibility and incident record

The manifest and all 9,924 boundary-logit rows were exported with runner commit
`541dd8e`. The first AEGBench decode stopped after encountering an incumbent
whose sub-picosecond interval gap was collapsed by 12-decimal endpoint rounding.
No logits were lost. Commit `0650535` preserves the exact identity boundary and
adds a defensive keep-incumbent fallback for any invalid trust region. The retry
reused the same cached logits and completed with three recorded invalid-trust-
region abstentions and four invalid-incumbent abstentions.

The final prediction SHA-256 values are
`5f49596cd7496df5f8dbe9858d24ce478cae01926f56fc926a0df8c19c3c0890`
for AudioGrounding and
`fc642d9e70c44068bdce895dcc2bfdfd40eaf02f4f5ed42f086c62e453040096`
for AEGBench. The final server test suite passed 99 tests.

CASTELLA was prepared but not executed: its public repository provides test
annotations and a YouTube acquisition script rather than redistributable raw
audio, and this server cannot reach YouTube. That is an input-availability
blocker, not a model score.
