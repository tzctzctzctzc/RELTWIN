# Benchmark-suite completion log

Branch: `codex/benchmark-suite-20260906`. Parent method commit: `cf57a33`.

| Benchmark | Public protocol | Status | Result |
|---|---:|---|---|
| AEGBench public v3 | 3,424 audio / 9,924 queries | completed | 44.803201 -> **45.276991** raw set-mIoU (`+0.473790`), 0 catastrophic regressions |
| UnAV100-subset public AMR | 77 audio / 100 queries | completed | 73.451288 -> **73.464610** mIoU (`+0.013323`), 0 catastrophic regressions |
| TUT Sound Events 2017 public AMR | 32 audio / 104 queries | completed | **24.762326 -> 24.762326** mIoU (all 104 abstained) |
| DESED public evaluation | 692 audio / 1,112 queries | completed | 56.592085 -> **57.000100** mIoU (`+0.408015`), 0 catastrophic regressions |
| LAT-Bench English TAG | 104 audio / 426 queries | full evaluation running | direct run stopped after processor revealed silent 600s truncation; label-free windowed run resumed |

Protocol mismatches against paper-reported counts are recorded explicitly and
will not be hidden by renaming a local expansion as an official table result.

The AEGBench number above is the complete current public-v3 expansion and is
not directly comparable to the paper's earlier 9,790-query table.  Its
audio-group bootstrap 95% interval for the NOVA gain is
`[+0.397863, +0.554112]`; 497 rows improve, 120 decline by less than 0.5 IoU,
and 9,307 tie.  Under the released Auto-AEG public-code metric, the same
predictions change mIoU from `40.820165` to `41.242504` (`+0.422339`).

UnAV uses the released 100-query AMR protocol, not the unreleased 997-query
subset reported in the SpotSound paper.  Primary scoring clips intervals to
the actual audio duration.  The unclipped audit score is
`73.374139 -> 73.387462`; both conventions give the same `+0.013323` paired
gain.  NOVA changes one row, improves it, and introduces no loss.

On TUT, the frozen router makes no boundary edit.  The low absolute score is
therefore inherited from the SpotSound-A incumbent and points to event
recall/multi-instance localization rather than a boundary-refinement failure:
61 of 104 rows contain multiple ground-truth intervals, while the incumbent
emits no multi-interval prediction.

On DESED, the locked adaptive-radius decoder switches 60 of 1,112 rows and
raises mIoU by `+0.408015` points.  The audio-group bootstrap 95% interval is
`[+0.259073, +0.567154]`; 53 rows improve, seven decline by less than 0.5 IoU,
and 1,052 tie.  The resulting `57.000100` remains below the SpotSound paper's
same-backbone SpotSound-A result of `57.8` by `0.799900` points, and below the
table-best SpotSound-Q result of `61.1` by `4.099900` points.  This is a valid
cross-benchmark improvement result, not a DESED SOTA claim.

The first LAT full run was intentionally stopped after 188 rows and is not a
reported result.  Its logs exposed the AF3 processor's hard 600-second cap;
continuing would make later events inaudible to the model.  The replacement
runner uses overlapping windows and the frozen SpotSound existence score for
label-free window selection.  The abandoned partial predictions remain on the
server only as diagnostic evidence.
