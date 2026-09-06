# Benchmark-suite completion log

Branch: `codex/benchmark-suite-20260906`. Parent method commit: `cf57a33`.

| Benchmark | Public protocol | Status | Result |
|---|---:|---|---|
| AEGBench public v3 | 3,424 audio / 9,924 queries | completed | 44.803201 -> **45.276991** raw set-mIoU (`+0.473790`), 0 catastrophic regressions |
| UnAV100-subset public AMR | 77 audio / 100 queries | data preparation | pending |
| TUT Sound Events 2017 public AMR | 104 queries | data preparation | pending |
| DESED public evaluation | grouped audio-label query sets | data preparation | pending |
| LAT-Bench English TAG | 104 audio / 426 queries | audio download and adapter validation | pending |

Protocol mismatches against paper-reported counts are recorded explicitly and
will not be hidden by renaming a local expansion as an official table result.

The AEGBench number above is the complete current public-v3 expansion and is
not directly comparable to the paper's earlier 9,790-query table.  Its
audio-group bootstrap 95% interval for the NOVA gain is
`[+0.397863, +0.554112]`; 497 rows improve, 120 decline by less than 0.5 IoU,
and 9,307 tie.
