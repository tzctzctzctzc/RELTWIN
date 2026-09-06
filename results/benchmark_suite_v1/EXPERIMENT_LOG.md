# Benchmark-suite completion log

Branch: `codex/benchmark-suite-20260906`. Parent method commit: `cf57a33`.

| Benchmark | Public protocol | Status | Result |
|---|---:|---|---|
| AEGBench public v3 | 3,424 audio / 9,924 queries | completed | 44.803201 -> **45.276991** raw set-mIoU (`+0.473790`), 0 catastrophic regressions |
| UnAV100-subset public AMR | 77 audio / 100 queries | completed | 73.451288 -> **73.464610** mIoU (`+0.013323`), 0 catastrophic regressions |
| TUT Sound Events 2017 public AMR | 32 audio / 104 queries | completed | **24.762326 -> 24.762326** mIoU (all 104 abstained) |
| DESED public evaluation | grouped audio-label query sets | data preparation | pending |
| LAT-Bench English TAG | 104 audio / 426 queries | corrected long-audio smoke | direct run stopped after processor revealed silent 600s truncation; label-free window selector pending |

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

The first LAT full run was intentionally stopped after 188 rows and is not a
reported result.  Its logs exposed the AF3 processor's hard 600-second cap;
continuing would make later events inaudible to the model.  The replacement
runner uses overlapping windows and the frozen SpotSound existence score for
label-free window selection.  The abandoned partial predictions remain on the
server only as diagnostic evidence.
