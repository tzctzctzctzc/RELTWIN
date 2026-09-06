# Adaptive boundary-radius iteration log

Run date: 2026-09-06. Branch: `codex/adaptive-radius-20260906`.

This experiment changes one factor only: the fixed 0.25-second boundary trust
region becomes `r_i = min(0.25 seconds, rho * incumbent_event_length_i)`. The
Boundary Utility model, bootstrap ensemble, features, margin (`0.02`), cached
logits, and incumbent predictions remain frozen. AudioGrounding-v2 and the local
AEGBench expansion are treated as post-hoc development sets because their
labels are used to select `rho`.

## Parameter iterations

All deltas are mIoU percentage points relative to the same rows' incumbent.
`Cat.` is the number of newly introduced per-row IoU drops of at least 0.5.

| Radius policy | AudioGrounding delta | AudioGrounding Cat. | AEGBench delta | AEGBench Cat. | Combined row-weighted delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed `0.25s` | -0.064800 | 1 | +0.783197 | 1 | +0.705826 | rejected: unsafe on both sets |
| `rho=0.25`, cap `0.25s` | +0.076699 | 0 | +0.060802 | 0 | +0.062253 | effective, conservative |
| `rho=0.5`, cap `0.25s` | +0.092063 | 0 | +0.357111 | 0 | +0.332914 | effective |
| `rho=0.625`, cap `0.25s` | +0.051659 | 0 | +0.473790 | 0 | +0.435253 | **selected and frozen** |
| `rho=0.75`, cap `0.25s` | -0.021689 | 0 | not run | not run | not applicable | rejected at AudioGrounding gate |

The search stopped after `rho=0.75` crossed back into negative transfer. The
selected `rho=0.625` is the largest tested passing scale and has the best
combined row-weighted delta among the safe tested configurations. No further
fine search was performed, limiting post-hoc overfitting.

## Frozen external-development result

With `rho=0.625`, AudioGrounding changes from `70.390059` to `70.441718`
(`+0.051659`, 28 wins / 956 ties / 13 losses, 41 switches, zero catastrophic
regressions). Its audio-group bootstrap 95% interval is
`[-0.135960, +0.227464]`.

AEGBench local expansion changes from `44.803201` to `45.276991`
(`+0.473790`, 497 wins / 9307 ties / 120 losses, 661 switches, zero catastrophic
regressions). Its audio-group bootstrap 95% interval is
`[+0.397863, +0.554112]`. This local positive-category expansion is diagnostic
and is not directly comparable to the later Auto-AEG leaderboard protocol.

Relative to the frozen absolute-radius decoder, adaptive radii eliminate both
observed catastrophic regressions and reverse AudioGrounding's negative
transfer, while retaining 60.5% of the AEGBench raw set-IoU gain.

## SpotSound and Clotho no-regression back-test

SpotSound full-400 was decoded after `rho` was frozen. It exactly reproduces
the current Boundary Utility result row by row: `59.328548` becomes `59.385511`
(`+0.056963`), with 36 switches and zero catastrophic regressions. Candidate
predictions, selected predictions, and switch decisions are all identical to
the fixed-radius run.

The equivalence is structural, not an estimate. The shortest incumbent event is
`0.5s` on SpotSound and `1.0s` on Clotho. Since every event satisfies
`0.625 * length >= 0.25s`, every adaptive radius is capped at exactly `0.25s`.
The candidate layers and deterministic decoder are therefore identical to the
existing full runs. Clotho consequently retains `86.854263 -> 86.855592`
(`+0.001329`), 9 switches, and zero catastrophic regressions without repeating
the redundant 6,649-row decode.

This change improves cross-benchmark short-event safety rather than adding
another point gain on SpotSound or Clotho. Their scores remain unchanged because
their incumbent events never enter the adaptive part of the radius function.

## Reproducibility

- Decoder implementation commit: `38a7ecc`.
- Frozen model SHA-256: `c129e88b274d9fe35593459b566bdd666e0d9c6c25ed4c4be7652a7059018158`.
- Frozen runtime config SHA-256: `c3a76b29a051fce661b8f132ef104a40c0d44a34019bb56a052c4cc3965f7174`.
- Selected prediction SHA-256: AudioGrounding
  `8766620849bcd9f4be6f470c6faaaad58d7d282fdaf761b0cbe17866a273614f7`,
  AEGBench `da410b790d19fe27925cb7e38091c9d77d64ae53568d31c68973eecc28eebea32`,
  SpotSound `6f0c12d11cefdd0e2b468252c434781ebed531de6c6d231d421b03880101d1bd4`.
- Strict merge checks found no missing rows, duplicates, duration mismatches, or
  source-index misalignment.
- Server test suite: 102 passed.

Large per-row predictions and cached logits remain under
`/root/autodl-tmp/SpotSound-ICASSP/autoresearch/06_experiments/runs/adaptive_radius_v1`.

