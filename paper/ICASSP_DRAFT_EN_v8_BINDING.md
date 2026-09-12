# RelTwin: Learning Query-Specific Windows from Co-Occurring Inverse Relations

Zhicheng Tang; Yuehan Zhang — Huazhong University of Science and Technology, Wuhan, China.

ICASSP 2027 author-review draft, 2026-09-13. Exported from the English manuscript; tables preserve the same evidence roles. The schematic is described in the methods rather than rasterized here.

## Abstract

Audio temporal grounding must distinguish a query's answer from other real event windows, not merely detect whether its sounds occur. We introduce RelTwin, a paired construction in which inverse relations coexist in one recording but identify different timestamp answers. Contrasting these answers augments sequence supervision without adding inference modules. A matched-runtime, equal-update comparison improves mIoU over SFT by 13.97 points on a class- and source-disjoint synthetic development split and by 1.02 points on the complete SpotSound-Bench, with grouped confidence intervals excluding zero. An advantage persists after uniformizing layout-specific silence. Across three seeds, candidate-supervised models reach 59.16 SpotSound mIoU, compared with the cited prior main-table best of 57.9. Extension controls do not establish separate benefits from exchange regularization or setwise continuation. The results support query-specific answer supervision and diagnostics requiring correct bidirectional localization, while leaving natural relational generalization open.

Keywords: audio temporal grounding; compositional reasoning; hard answer negatives; paired supervision.

## Introduction

Audio temporal grounding maps a natural-language query to the intervals that answer it, beyond fixed event classes [Text-to-Audio Grounding](https://arxiv.org/abs/2102.11474), [Language-based Audio Moment Retrieval](https://arxiv.org/abs/2409.15672). SpotSound equips an audio-language model with interleaved timestamps and training for absent events [SpotSound, Table 3](https://arxiv.org/html/2604.13023v2). We ask a complementary question: when several real windows contain the same event categories, does the model select the one specified by the query's relation?

Consider a recording containing “dog then bell” and, later, “bell then dog.” Both descriptions are true. Reversing the query should change the answer window while the audio and event vocabulary stay fixed. A model can fail without hallucinating a sound: it returns a real occurrence answering the wrong relation. We call this a *query–window binding error*. Correct event recognition alone does not determine which occurrence to return.

T-CLAP uses concatenated sounds, reversed descriptions, and temporal contrastive training [T-CLAP](https://arxiv.org/html/2404.17806v1); CompA studies event order and attribute binding through paired audio–caption matching [CompA](https://arxiv.org/html/2310.08753v3). CoSTALA develops hierarchical spatio-temporal alignment [CoSTALA](https://arxiv.org/html/2608.24374v1), and SHINE uses compositional hard negative queries in video grounding [SHINE](https://arxiv.org/abs/2407.05118). Our specific setting is *two inverse descriptions simultaneously true in one recording, but identifying different local timestamp answers*. The wrong window is an answer negative, not evidence that the query is globally false: supervision must distinguish query-specific answers among real windows.

We introduce RelTwin: the same sound excerpts form two opposite-order windows, and each inverse query is scored against both timestamp answers. Ordinary candidate cross-entropy augments correct-answer sequence supervision. Our contribution is this locally confusable answer construction and its implementation in generative grounding, rather than a new contrastive principle. Inference remains ordinary timestamp generation, without a candidate scorer or target-window access.

We test whether this supervision improves over same-data SFT, whether more elaborate objectives are necessary, and whether the advantage survives a timing-cue stress test. Paired diagnostics check that answers change *correctly*, not merely that outputs differ. Public grounding results then connect this controlled behavior to practical localization performance.

## Query-Specific Window Learning

### Two true relations, two different answers

Let $x$ be a recording, $q$ a query, and $G$ its target interval set. RelTwin reuses two ESC-50 excerpts [ESC-50](https://github.com/karolpiczak/ESC-50), $A$ and $B$, to form two windows $W_+=(A\!\rightarrow\!B)$ and $W_-=(B\!\rightarrow\!A)$ in the same recording. Each window spans the *entire ordered sequence*, not only one constituent event. For the two inverse queries, the answer-label matrix is

$$

Y=\begin{array}{c|cc}
 &W_+&W_-\\ \hline
q_+&1&0\\
q_-&0&1
\end{array}.

$$

Recording-level presence labels mark both queries true but do not specify Eq. 1. The distinction concerns supervision, not the capabilities of globally pretrained models.

Each recording has two templates, “followed by” and “and then,” in both directions. Category queries target all occurrences of $A$ or $B$ for rehearsal. Alternating the earlier relation controls a constant-position preference; excerpt reuse fixes source material within a pair. Layout-specific silence remains a potential shortcut, tested in Section 4.3.

### Contrast timestamp answers during training

For the timestamp token sequence $y(S)$ describing interval set $S$, define its mean token log-likelihood

$$

s_\theta(x,q,S)=\frac{1}{|y(S)|}\sum_t\log P_\theta(y_t\mid x,q,y_{<t}).

$$

Candidates are $C=(W_+,W_-)$. Their normalized preferences are $p_q(j)=\operatorname{softmax}_j(s_\theta(x,q,C_j)/\tau)$, not calibrated event-presence probabilities. The paired candidate loss is

$$

\mathcal L_{\rm cand}=-\tfrac12[\log p_{q_+}(1)+\log p_{q_-}(2)].

$$

This standard candidate cross-entropy compares the correct answer with the other *present* window. We add correct-answer sequence supervision and category-query rehearsal:

$$

\mathcal L=\mathcal L_{\rm seq}+\lambda\mathcal L_{\rm cand}
+\beta\mathcal L_{\rm replay}.

$$

SFT uses identical data and rehearsal but omits $\mathcal L_{\rm cand}$. Equation 4, *RelTwin-Cand*, is the existing no-exchange configuration, not a newly tuned checkpoint. Inference generates intervals without access to $W_+$ or $W_-$.

### Testing whether extra objectives are needed

Relation-Boundary Exchange Equivariance (RBEE) adds $\lambda\operatorname{JS}(p_{q_+},\Pi p_{q_-})$, where $\Pi$ swaps candidate indices. This enforces exchange consistency, not architectural equivariance. Perfect candidate labels already satisfy this exchange.

Setwise Preference Optimization (SetPO) adds 64 updates with six answers: both windows, their expanded versions, their union, and one spanning interval. Soft targets average set-IoU and symmetric best-match interval F1 (not one-to-one event F1). We compare it with RBEE continued for 64 updates to isolate the objective from additional training.

### Check both localization and relation selection

For interval unions, $\operatorname{IoU}_{\mathrm{set}}(S,G)=|S\cap G|/|S\cup G|$. PairAcc requires both inverse-query predictions to achieve set-IoU $\geq0.5$. We additionally require each to overlap its own target strictly more than the opposite target, yielding JointPairAcc. Swap error means that at least one query violates this strict preference, including ties.

Returning the union of two disjoint equal-length targets for *both* queries gives IoU $0.5$ each: PairAcc passes but JointPairAcc fails. This task-specific diagnostic follows paired compositional evaluation [Winoground](https://arxiv.org/abs/2204.03162); it complements standard IoU rather than replacing it.

## Experimental Setup

**Data and split roles.**
Adaptation uses 40 ESC-50 classes and 256 constructed recordings: 1,024 relation queries (512 pairs) and 512 rehearsal queries. The remaining 10 classes provide 80 recordings, 320 queries, and 160 inverse pairs. Adaptation/development classes and original Freesound IDs are disjoint, without implying absence from backbone pretraining. This relation split has informed development, not independent natural-scene validation. Public evaluation uses all 400 SpotSound queries over 387 audios. These results have also been inspected during development; they are not a pristine selection holdout.

**Matched adaptation.**
All variants start from SpotSound-A on Audio Flamingo 3, updating 20,185,088 adapter parameters. SFT/Cand/RBEE use 256 AdamW updates, zero weight decay, learning rate $5\times10^{-6}$, gradient clipping 1.0, $\tau=\lambda=1$, and $\beta=0.5$. Cand/RBEE have seeds 0/1/2. Newly trained SFT seed 0 matches Cand seed 0 in runtime, code, initial weights, data order, and configuration; only the objective and its implementation differ. This primary contrast controls updates, not FLOPs: Cand scores four query--answer combinations without gradients then recomputes weighted gradients; SFT backpropagates two positive answers. Both add rehearsal.

RBEE continuation and SetPO add 64 updates at $2\times10^{-6}$ from each RBEE parent. SetPO uses method weight 0.5, target temperature 0.15, exchange weight 1, rehearsal weight 0.5, and setwise rehearsal every fourth update. We use final checkpoints without seed selection. Historical SFT, RBEE, and SetPO share current inference but retain historical training-runtime differences; their contrasts are separate from matched SFT–Cand.

**Uncertainty and reproducibility.**
We verify row alignment, audio identity, duration, and recomputed IoU. Paired bootstrap uses 20,000 audio-group draws, keeping inverse queries and templates together. To address excerpt reuse, relation intervals additionally resample 39 original-source connected groups (largest: 18 audios). CIs condition on observed seeds and do not correct development selection. Code, inputs, weights, and environment are hashed before inference.

### Table 1. Public grounding results (%)

Literature values are from SpotSound Table 3, not its ablations. RelTwin rows are three-seed current-harness means. The archived full system is a distinct dataset-specific pipeline; Clotho is official plus refinement, not RelTwin transfer. The final row compares with each metric's prior main-table best. Bold denotes the highest listed point estimate, not significance.

| Method | SpotSound mIoU | R1@.3 | R1@.5 | Clotho mIoU | R1@.3 | R1@.5 |
|---|---|---|---|---|---|---|
| WTATG | 38.4 | 53.0 | 34.5 | 9.1 | 12.1 | 6.3 |
| AM-DETR | 19.5 | 25.5 | 15.0 | 80.9 | 89.8 | 88.0 |
| Gemini-2.5-Flash | 25.7 | 34.8 | 27.8 | 36.9 | 45.7 | 33.7 |
| Gemini-2.5-Pro | 19.7 | 20.0 | 17.3 | 32.5 | 40.4 | 32.5 |
| Kimi-Audio | 3.2 | 3.0 | 1.3 | 0.9 | 0.7 | 0.1 |
| Qwen2-Audio | 4.4 | 5.5 | 1.8 | 5.7 | 6.1 | 1.3 |
| Audio Flamingo 3 | 7.6 | 8.8 | 2.8 | 22.6 | 32.9 | 21.8 |
| TimeAudio | 10.2 | 7.8 | 1.8 | 28.6 | 39.5 | 24.9 |
| SpotSound-Q | 49.9 | 65.0 | 51.3 | 85.4 | **93.6** | **91.2** |
| SpotSound-A | 57.9 | 76.5 | 59.5 | 85.6 | 93.4 | 91.0 |
| RelTwin-Cand (3 seeds) | 59.16 | 77.17 | 62.08 | -- | -- | -- |
| RelTwin-RBEE (3 seeds) | 59.29 | 76.92 | 62.33 | -- | -- | -- |
| Archived full system | **59.39** | **79.25** | **62.50** | **86.86** | 93.49 | 90.84 |
| System $\Delta$ vs. prior best | +1.49 | +2.75 | +3.00 | +1.26 | $-0.11$ | $-0.36$ |

### Table 2. Unified controlled evaluation (%)

Current SFT and Cand seed 0 share the training runtime. Lower rows use all three observed seeds; ± is sample standard deviation, not CI. Historical models retain the training-runtime caveat described in the setup.

| Method | SpotSound mIoU | Relation mIoU | JointPairAcc |
|---|---|---|---|
| Official | 58.42 | 37.64 | 0.63 |
| Historical SFT, s0 | 58.76 | 75.34 | 55.63 |
| Current SFT, s0 | 58.42 | 74.58 | 53.13 |
| Cand, s0 | 59.43 | 88.55 | 84.38 |
| Cand, 3 seeds | $59.16\pm0.26$ | $87.05\pm1.43$ | 81.46 |
| RBEE, 3 seeds | $59.29\pm0.19$ | $86.78\pm1.11$ | 81.67 |
| RBEE +64 updates | $59.18\pm0.13$ | $89.04\pm0.13$ | 85.63 |
| SetPO +64 updates | $59.20\pm0.19$ | $89.81\pm0.42$ | 86.67 |

### Table 3. Extension contrasts

Intervals use original-source connected groups. Neither contrast establishes an independent extension benefit.

| Contrast | Metric | Delta (points) | 95% source-group CI |
|---|---|---|---|
| RBEE $-$ Cand | mIoU | $-0.27$ | $[-0.70,0.06]$ |
|  | JointPairAcc | $+0.21$ | $[-0.88,0.96]$ |
| SetPO $-$ continue | mIoU | $+0.77$ | $[-0.05,1.96]$ |
|  | JointPairAcc | $+1.04$ | $[-1.24,4.76]$ |

## Results and Analysis

### Candidate supervision beyond the same-data SFT control

The matched-runtime comparison in Table 2 improves relation mIoU from 74.58 to 88.55 ($+13.97$ points), with source-group 95% CI $[9.84,17.51]$. JointPairAcc rises from 53.13 to 84.38 ($+31.25$ points; CI $[22.35,40.50]$), and Swap error falls from 40.00 to 7.50. On SpotSound, SFT scores 58.42 and Cand 59.43: $+1.02$ mIoU points with audio-group CI $[0.23,1.81]$. This supports an advantage over this fixed same-data, same-update SFT run, not over every SFT configuration. Across 400 public queries, Cand wins on 92, ties on 239, and loses on 69; two losses are at least 0.5 IoU. Thus the gain is aggregate, not per-example dominance.

The behavior analysis clarifies what changes. SFT returns exactly the same answer for 37/160 inverse pairs (22 audios); Cand does so for only 1/160. Of those 37 SFT-collapse pairs, 27 become jointly correct under Cand. Over all pairs, 51 previously incorrect pairs become jointly correct and one previously correct pair is lost. Output diversity alone is not the criterion: Cand still fails JointPairAcc on 25/160 pairs. These are descriptive decompositions of the same development split, not additional independent tests.

### Are exchange and setwise extensions necessary?

Across three seeds, Cand and RBEE reach 87.05 and 86.78 relation mIoU, and 81.46 and 81.67 JointPairAcc. Both JS-increment intervals cross zero (Table 3). This does not prove equivalence, but supports retaining the simpler candidate-supervised design rather than claiming an independent exchange benefit.

SetPO improves over its RBEE parents by 3.04 mIoU points, but continuing RBEE alone recovers 2.27 points. Its equal-update advantage is 0.77 mIoU and 1.04 JointPairAcc points, with intervals crossing zero; the SpotSound difference is 0.019 points. The parent comparison therefore establishes continuation-stage improvement, not an isolated setwise-objective gain.

### Does the advantage survive removing a timing cue?

The original AB-first and BA-first layouts use different silence patterns. We recompose all 80 development audios with a common pattern, preserving exact event waveforms, order, and queries: 1.5 s lead-in, 0.25 s within-window gap, 3 s between windows, and 2 s tail. Frozen SFT/Cand seed 0 models evaluate one template (160 queries), without tuning or retraining. Forty AB-first audios remain waveform-identical and reproduce every prediction, providing an execution control.

On the 40 changed BA-first audios, JointPairAcc is 67.50 for SFT and 87.50 for Cand: $+20.00$ points, source-group CI $[2.94,33.33]$. The original gap on these audios was 42.50 points (42.50 versus 85.00). Uniform timing helps SFT substantially and narrows the gap, while Cand retains an advantage. Across all 160 queries, uniform-timing mIoU is 81.34 versus 90.65. Thus this specific timing cue does not fully explain the result. This is a post-development stress test, not independent natural-audio validation.

### Public scores answer a different question

Table 1 compares with published *main-table* results. Cand and RBEE average 59.16 and 59.29 SpotSound mIoU versus the cited 57.9. Our official re-evaluation already scores 58.42, so adaptation cannot explain the entire literature gap. Cand exceeds this official run by 0.75 points across three seeds, with audio-group CI $[-0.24,1.74]$. Unlike the matched SFT contrast, this difference is not statistically established.

The archived SpotSound system additionally routes official/SFT/SetPO answers using keep/drop verification fitted on LongNeedle-ESC50, then applies local Boundary Utility refinement. Its 59.39 mIoU exceeds the cited best by 1.49 points; the recall gains are 2.75 and 3.00 points. Refinement alone changes 59.328548 to 59.385511. We do not mix archived and current predictions to estimate component effects.

Clotho's 86.86 mIoU uses official plus refinement, without RelTwin; its incumbent already scores 86.854263. Its recalls are slightly below the cited best. This is system context, not relation-transfer evidence. UnAV is excluded from fair ranking because its public 100-query manifest remains unreconciled with the source protocol.

### Evidence boundaries

Simple constructed sequences do not establish robustness to natural, overlapping, nested, or long-range relations. Repeated excerpts and fixed wording remain biases after timing control. Class/source separation and source-group bootstrap do not remove these biases or development selection.

Historical SFT changes on 49/400 SpotSound queries under re-evaluation despite identical input token counts, motivating our current cohort and matched-runtime control. The primary SFT–Cand contrast remains one seed, at equal updates rather than compute. Independent natural-relation validation is the principal remaining generalization requirement.

## Conclusion

RelTwin learns which real window answers a query through co-occurring inverse relations and explicit answer negatives. Same-data training controls and paired diagnostics support candidate supervision, whose advantage persists after a targeted timing intervention. Exchange and setwise extensions show no established independent gain. This supervision design improves query-specific localization without adding inference modules; natural relational generalization remains open.

## AI assistance disclosure

OpenAI Codex assisted with initial drafting and translation of all sections, the schematic, LaTeX preparation, and analysis code.

## References

- [Text-to-Audio Grounding](https://arxiv.org/abs/2102.11474)
- [Language-based Audio Moment Retrieval](https://arxiv.org/abs/2409.15672)
- [SpotSound, Table 3](https://arxiv.org/html/2604.13023v2)
- [T-CLAP](https://arxiv.org/html/2404.17806v1)
- [CompA](https://arxiv.org/html/2310.08753v3)
- [CoSTALA](https://arxiv.org/html/2608.24374v1)
- [SHINE](https://arxiv.org/abs/2407.05118)
- [ESC-50](https://github.com/karolpiczak/ESC-50)
- [Winoground](https://arxiv.org/abs/2204.03162)
