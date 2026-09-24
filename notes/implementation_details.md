# Implementation details retained outside the four-page body

## SpotSound-A matched candidate-supervision ablation (Table 2)

Paired SFT performs two positive-answer backpropagations. Candidate supervision additionally computes four gradient-free answer scores and uses weighted gradient recomputation. This implementation detail was moved from Section 3.1 to make room for analysis; the training procedure is unchanged. Initialization, data/order, update count, loss switches, optimizer, LoRA settings, and accumulated queries remain in the paper.

## Case provenance and pending public-benchmark examples

The current case analysis uses the frozen controlled inverse-query example in Figure 1, with timestamps and IoUs from `figures/overview_data.tex`. It compares paired SFT and candidate supervision, not the full-framework public-benchmark checkpoints. It is not relabeled as a public benchmark example.

A public-benchmark success/failure pair is pending the corresponding ground truth and aligned official/full-RelTwin predictions. Older NOVA or no-exchange outputs must not be substituted for the main table's full-RelTwin runs. The TimeAudio 54.33% swap error cited in the limitations paragraph is the full-RelTwin aggregate in current Table 2, not a newly diagnosed per-example failure category.
