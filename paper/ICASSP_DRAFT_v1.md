# NOVA: Learning When to Refine Audio Temporal Boundaries

Working draft for ICASSP — 2026-09-08. This is a content draft, not a camera-ready layout. No new experiments were run for this document.

## Abstract

Refining a temporal prediction is useful only when the edit improves localization over the original answer. For audio temporal grounding, dense acoustic evidence can suggest sharper boundaries without establishing that moving them will improve temporal overlap. We present a selective boundary-refinement extension to NOVA that learns the value of an edit relative to an existing prediction. A lightweight utility model uses frozen occupancy and boundary features to predict changes in set intersection-over-union. A constrained dynamic program selects compatible local edits, while the identity action explicitly preserves the original prediction. Edits are accepted only when a lower bootstrap utility quantile exceeds a fixed margin; an event-duration-dependent radius limits displacement for short events. The resulting system achieves 59.39% mIoU on SpotSound-Bench and 86.86% on Clotho-Moment, exceeding the corresponding best results in the SpotSound main table. Paired evaluations show small boundary-refinement gains on these benchmarks and a 0.41-point gain on DESED, with no observed per-query IoU drops of at least 0.5 on these three evaluations. These results motivate evaluating refinement through both localization gain and the damage introduced by editing strong predictions.

**Index Terms—** Audio temporal grounding, audio-language models, selective refinement, boundary utility.

## 1. Introduction

Audio temporal grounding localizes the time intervals described by a natural-language query. Recent audio-language models provide a strong starting point: SpotSound introduces timestamp-interleaved audio representations and explicit training for absent events, while audio moment retrieval extends query-based localization to untrimmed recordings. These advances make it useful to study not only how a model produces an interval, but also how that interval should be revised after generation. [SpotSound](https://arxiv.org/html/2604.13023v2), [Language-based Audio Moment Retrieval](https://arxiv.org/abs/2409.15672).

Our focus is the decision to revise an existing answer. An acoustic transition is evidence about an event boundary, but it is not itself evidence that a particular edit improves localization over the current prediction. A local correction can remove an irrelevant tail or truncate a correctly localized event. Its consequences also depend on event duration: shifting a short event by a fixed number of milliseconds changes a larger fraction of its temporal support. A useful refinement mechanism must therefore answer two questions together: which edit is promising, and whether the evidence is strong enough to replace the original prediction.

Boundary refinement and duration-aware correction are established directions in temporal grounding. TimeRefine predicts successive offsets to a coarse video interval, and TimePLE introduces duration-dependent bounded correction within a temporal representation framework. Our contribution is not the general idea of refining boundaries or adapting a search radius. We instead formulate local audio refinement as an incumbent-relative utility decision, with retaining the incumbent represented explicitly in the action space. This formulation targets the distinction between proposing a plausible boundary and predicting that an edit is beneficial. [TimeRefine](https://arxiv.org/abs/2412.09601), [TimePLE](https://arxiv.org/html/2607.23951v1).

We implement this formulation as a boundary-refinement extension to NOVA. Given an existing interval set, frozen acoustic heads provide occupancy, onset, and offset evidence. A lightweight supervised model learns the set-IoU change caused by individual legal edits, using acoustic-feature differences and edit geometry. Dynamic programming selects compatible edits under an additive learned-utility surrogate, preserving interval count, temporal order, and non-overlap. The original prediction remains an exact identity action. A lower quantile of utility predictions from audio-group bootstrap models must exceed a fixed margin before the selected edit is applied. We additionally cap displacement by event duration to constrain edits to short events. This is an empirical selection rule, not a distribution-free guarantee of non-degradation.

Our evaluation separates complete-system accuracy from the incremental effect of refinement. The resulting system obtains 59.39% and 86.86% mIoU on SpotSound-Bench and Clotho-Moment, respectively. Relative to their fixed incumbents, boundary refinement adds 0.057 and 0.001 points; on DESED it adds 0.408 points. Development experiments on AudioGrounding and AEGBench show that duration-dependent radii remove two observed severe regressions introduced by fixed-radius refinement. These development findings are not independent test evidence. Together, the experiments support a focused contribution: learning incumbent-relative edit utility, combining it with explicit abstention and constrained interval selection, and evaluating both refinement gains and introduced regressions. The method does not add missed events or repair an incorrect distant retrieval window.

## Table 1. Main comparison draft

**Caption:** Audio temporal grounding mIoU (%). Prior-system results are transcribed from SpotSound Table 3, not its ablation tables. The final row reports numerical differences from the best published value in each column; it does not measure the gain attributable to boundary refinement. Our row is an open-extra-data system configuration and uses different fixed incumbents across benchmarks. Columns marked with a dagger are reference comparisons with unresolved manifest equivalence; the double dagger additionally marks development-data use. Boldface is restricted to the two primary comparison columns. No statistical significance is implied.

| Method | SpotSound-Bench | Clotho-Moment | UnAV-100 subset † | AudioGrounding-v2 †‡ |
|---|---:|---:|---:|---:|
| WTATG | 38.4 | 9.1 | 38.4 | 51.4 |
| AM-DETR | 19.5 | 80.9 | 42.8 | 30.2 |
| Gemini-2.5-Flash | 25.7 | 36.9 | 35.6 | 37.1 |
| TimeAudio | 10.2 | 28.6 | 16.0 | 67.4 |
| SpotSound-Q | 49.9 | 85.4 | 72.4 | 67.8 |
| SpotSound-A | 57.9 | 85.6 | 69.8 | 70.3 |
| NOVA + selective boundary refinement (ours) | **59.39** | **86.86** | 73.46 † | 70.47 †‡ |
| Difference from prior main-table best (points) | +1.49 | +1.26 | +1.06 † | +0.17 †‡ |

Published rows: [SpotSound, Table 3](https://arxiv.org/html/2604.13023v2#S4).

**Sample counts for our evaluations:** SpotSound 400; Clotho 6,649; UnAV 100 queries over 77 audio files; AudioGrounding 997 queries over 483 queried audio groups (492 source records).

**† Protocol qualification.** The complete upstream AudioGrounding and AMR UnAV releases were evaluated. SpotSound Table 2 assigns AudioGrounding 70 audio/100 queries and UnAV 492 audio/997 queries, whereas the released sources associate the 997-query set with AudioGrounding and the 100-query set with UnAV. No separate SpotSound manifest was found to resolve this inconsistency. These numerical comparisons do not establish exact sample-level equivalence. Our primary scores clip annotations and predictions to decoded audio duration; the original paper's endpoint convention has not been confirmed. Without clipping, our scores are 70.441718 and 73.387462, respectively. For submission, separate these columns into a reference/development panel or move them out of the primary table unless comparability is resolved.

**‡ Development use.** AudioGrounding labels were used to choose the duration-radius coefficient. Rerunning inference on the released test file does not restore its independence. It must not be presented as an untouched test or as independent SOTA evidence. AEGBench was also used for this selection.

**Training and system scope.** The SpotSound incumbent is the existing E004 NOVA system, including prior relation-aware adaptation and verification; Clotho, AudioGrounding, and UnAV use the official SpotSound-A incumbent. The refinement module uses frozen SetPO/SpanTool evidence and a separately trained utility model. The comparison is neither same-training-data nor an evaluation of a single identical incumbent checkpoint on every benchmark. “Frozen-backbone refinement” is accurate; “entirely training-free” is not.

**Interpretation.** The supported claim is that the two primary scores exceed the corresponding best entries in the cited SpotSound main table. This is not an exhaustive certification of all contemporaneous leaderboards. The table deliberately follows the main-table comparison requested for this draft; it does not imply that no higher dataset-specific ablation score has been published.

## Internal author notes — not manuscript prose

### Why this story and title

主线：强预测已经存在，精细声学证据并不自动意味着应该修改；我们学习“相对于原答案，修改是否值得”，把保持原答案作为正式动作，并约束短事件的修改范围。不要把故事写成“首次发现短事件对边界误差敏感”或“首次提出动态半径”。NOVA 沿用项目既有名称 Necessity-Oriented Verification over Audio；新增模块称 selective boundary refinement / Boundary Utility，不为 NOVA 重新杜撰英文全称。

### Mandatory attribution companion

This table is an ablation/attribution companion, not a replacement for comparison to published systems. It should remain in the experimental section even if the main table is shortened.

| Evaluation | Fixed incumbent mIoU | With refinement | Paired gain (points) | New catastrophic regressions |
|---|---:|---:|---:|---:|
| SpotSound-Bench | 59.328548 | 59.385511 | +0.056963 | 0 |
| Clotho-Moment | 86.854263 | 86.855592 | +0.001329 | 0 |
| DESED, released evaluation assembly | 56.592085 | 57.000100 | +0.408015 | 0 |
| UnAV, released AMR subset | 73.451288 | 73.464610 | +0.013323 | 0 |
| AudioGrounding, development; clipped | 70.418138 | 70.469831 | +0.051693 | 0 |

A catastrophic regression is a per-query set-IoU decrease of at least 0.5 on the [0,1] scale. Zero observed catastrophic regressions does not mean zero losses or guaranteed safety. SpotSound has 20 wins / 365 ties / 15 losses, Clotho 5 / 6,640 / 4, and DESED 53 / 1,052 / 7. SpotSound's paired 95% audio-group bootstrap interval is [-0.114667, +0.236260] points; DESED's is [+0.259073, +0.567154]. UnAV changes only one query. The large differences from paper-reported scores cannot be attributed entirely to this boundary module.

### Mechanism and evidence limits

The deployed decoder is Boundary Utility, not the earlier posterior-IoU Dinkelbach proposal. DP optimizes an additive learned utility, not exact expected multi-interval set-IoU. The deployed radius is `min(0.25 seconds, 0.625 * incumbent_event_duration)`, the utility ridge penalty is 0.1, and the acceptance margin is 0.02 in IoU units. The 5th percentile over 200 fitted bootstrap utility models is an ensemble decision statistic, not a certified 95% lower confidence bound on realized test gain.

Every SpotSound and Clotho incumbent interval hits the 0.25-second cap. Their adaptive-radius predictions are identical to fixed-radius predictions. Therefore adaptive radii contribute no additional improvement on those two datasets. The short-event mechanism evidence comes from AudioGrounding/AEGBench development experiments and requires independent confirmation. Existing SpotSound public aggregates also informed earlier method iterations; do not describe the whole project as a pristine, never-inspected confirmatory evaluation.

DESED remains below the paper main-table best of 61.1 despite its paired improvement. TUT is unchanged at 24.762326; LAT declines from 15.146085 to 15.141936. Keep these outcomes in the reproducibility record and discuss multi-event recall and long-audio retrieval as limits. Choosing a grounding-focused main table does not justify claiming universal gains across the full evaluated suite.

### Before submission

- Keep one compact main comparison plus one paired mechanism table/figure; do not substitute a leaderboard gap for a module ablation.
- Compare the same candidate space with no utility gate, mean-utility selection, and the bootstrap gate before claiming the gate itself is responsible for improved reliability. Those counterfactual results are not supplied by this draft.
- Resolve the two upstream-manifest comparisons or keep them explicitly reference/development-only. Audit all training/calibration/evaluation audio-group overlaps before claiming independent generalization.
- Archive all eight benchmark outcomes; disclose tuning history, extra data, and benchmark-specific incumbent choice.
- Verify the target ICASSP year's official template and page policy when converting this content draft into a submission. No year-specific formatting compliance is claimed here.

### Local evidence

- `README.md`: existing system components, adaptation data, public-iteration caveats, and exact attribution.
- `scripts/boundary_utility.py`: utility fitting, identity action, DP surrogate, and acceptance rule.
- `results/nova_boundary_utility_v1/spotsound_full400_report.json` and `clotho_full6649_report.json`.
- `results/adaptive_radius_v1/EXPERIMENT_LOG.md`: post-hoc radius choice and primary-benchmark equivalence.
- `results/official_protocol_rerun_20260907/EXPERIMENT_LOG.md`: released-source reruns, endpoint sensitivity, and manifest mismatch.
- `results/benchmark_suite_v1/EXPERIMENT_LOG.md`: DESED, TUT, AEGBench, UnAV, and LAT results.

All scores were checked against these records on 2026-09-08. The baseline paper is SpotSound v2. Related-work anchors were reopened from their primary arXiv pages for this draft; this writing pass is not a new exhaustive novelty search.
