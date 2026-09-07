# AudioGrounding and UnAV official-release rerun

Run completed on 2026-09-08 on branch
`codex/official-ag-unav-20260907`. The server test suite passed all 125 tests.
Both incumbent generation and NOVA boundary extraction were rerun from scratch;
all 1,097 rows completed with valid alignment.

## Main comparison

The table compares the current NOVA result directly with the best result in
SpotSound Table 3, not with our own incumbent. Scores use set-IoU after clipping
intervals to the actual WAV duration.

| Benchmark | Released protocol | Previous best in SpotSound main table | Current NOVA | Difference from previous best | R1@.3 | R1@.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AudioGrounding-v2 | 492 source records / 997 queries (483 queried audio) | 70.3 mIoU, SpotSound-A | **70.469831** | **+0.169831** | 89.167503 | 76.328987 |
| UnAV-100 subset | 77 audio / 100 queries | 72.4 mIoU, SpotSound-Q | **73.464610** | **+1.064610** | 89.000000 | 80.000000 |

Under the complete released-source protocols, both point estimates exceed the
corresponding prior best mIoU in the SpotSound main table. AudioGrounding's
R1@.3 is below the paper's 90.1, while its mIoU and R1@.5 exceed 70.3 and 74.8.
UnAV exceeds all three paper values: 72.4 mIoU, 88.0 R1@.3, and 74.0 R1@.5.

## Paired effect of NOVA

| Benchmark | Frozen incumbent | Current NOVA | Paired delta | Audio-group bootstrap 95% CI | Win / tie / loss | Switches | Catastrophic regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AudioGrounding-v2 | 70.418138 | **70.469831** | +0.051693 | [-0.135860, +0.227569] | 28 / 956 / 13 | 41 | 0 |
| UnAV-100 subset | 73.451288 | **73.464610** | +0.013323 | [0.000000, +0.042071] | 1 / 99 / 0 | 1 | 0 |

These paired deltas explain what the new boundary module changes, whereas the
main table above is the correct comparison for a benchmark claim. The
AudioGrounding confidence interval includes zero, so its gain over our
incumbent is a positive point estimate rather than a statistically resolved
improvement. UnAV changes only one query; its gain is correspondingly small.

## Endpoint audit

AudioGrounding contains 132 annotation rows that exceed the decoded WAV end by
at most 73.5 ms. UnAV contains four such rows, including one interval ending
1.978 s after its WAV. Keeping the released endpoints without clipping gives:

| Benchmark | Incumbent | Current NOVA | Paired delta | Difference from paper best |
| --- | ---: | ---: | ---: | ---: |
| AudioGrounding-v2 | 70.390059 | **70.441718** | +0.051659 | +0.141718 |
| UnAV-100 subset | 73.374139 | **73.387462** | +0.013323 | +0.987462 |

The conclusion is unchanged under either convention.

## Protocol-cardinality correction

SpotSound Table 2 lists AudioGrounding as 70 audio/100 queries and UnAV as 492
audio/997 queries. The public source files establish the reverse association:
the official AudioGrounding-v2 `test.json` has 492 records and 997 phrase
queries over 483 records containing queried phrases, while the AMR authors
release UnAV as 100 queries over 77 available audio files. The original
SpotSound repository does not release a separate
benchmark manifest or evaluation script that resolves this inconsistency.

Accordingly, these are complete upstream official-release evaluations and can
be compared with SpotSound Table 3 only with the Table 2 count inconsistency
footnoted. They should not be described as reproducing a distinct, unavailable
70/100 AudioGrounding split or a distinct 492/997 UnAV split.

## Reproducibility

- Code commit used for inference and decoding:
  `671fba7473c16f19f721e48e79e6b65b7770c874`.
- AudioGrounding released source SHA-256:
  `426378178ecea5f87e4f927431bb987a598bcc59c7c6113ec268970c7364d4661`.
- AudioGrounding NOVA prediction SHA-256:
  `d1878375cc604466deb18f8b61823d846aad0501279a504e372a263f9d335851`.
- UnAV released source SHA-256:
  `e7e471117f658108e92adb313238e2609234ef5f85c9ddf26d6c09d14d10ffdec`.
- UnAV NOVA prediction SHA-256:
  `37e70046e159f1d0156276b46edeab7515c0e5147ebd3575a99567368f18d1af`.
- Repeated incumbent predictions are identical to the prior cache on every
  semantic and scoring field. NOVA selections are also identical; two
  AudioGrounding rows differ only in insignificant serialized floating-point
  `delta_iou` values caused by the GPU export environment.
