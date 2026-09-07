# AudioGrounding and UnAV released-protocol rerun

This run evaluates the frozen SpotSound-A incumbent and the frozen NOVA
adaptive-radius boundary decoder on the complete upstream public releases. No
benchmark label is used to tune the model, decoder, margin, or radius policy in
this rerun.

## Frozen method

- Code parent: `671fba7473c16f19f721e48e79e6b65b7770c874`.
- Result branch: `codex/official-ag-unav-20260907`.
- Incumbent: released SpotSound Audio Flamingo 3 adapter.
- Boundary features: SetPO seed-1 adapter and the frozen SpanTool checkpoint.
- Decoder: Boundary Utility with `r_i = min(0.25s, 0.625 * L_i)` and decision
  margin `0.02`.
- Evaluation seed: `20260903`; audio-group bootstrap: 10,000 samples.
- Primary scoring clips ground-truth and prediction intervals to the decoded
  WAV duration. An unclipped released-endpoint score is retained as an audit.

## Released data

AudioGrounding-v2 uses all 997 phrase queries from 492 records in the official
`test.json`. UnAV uses all 100 queries over 77 audio files in the AMR authors'
`unav100-subset_test_release.jsonl`.

SpotSound Table 2 instead reports AudioGrounding as 70 audio/100 queries and
UnAV as 492 audio/997 queries. Those counts conflict with the named upstream
releases and line up with the opposite datasets: 492/997 is the complete
AudioGrounding-v2 test release, while the released UnAV AMR subset is 77/100.
The rerun therefore records the upstream release identities and treats the
Table 2 cardinalities as a paper metadata inconsistency. The Table 3 method
scores remain the comparison target, with this inconsistency disclosed.

Source references:

- SpotSound paper: <https://arxiv.org/html/2604.13023v2>
- AudioGrounding official repository: <https://github.com/wsntxxn/TextToAudioGrounding>
- AudioGrounding-v2 release: <https://doi.org/10.5281/zenodo.7269161>
- Language-based Audio Moment Retrieval project: <https://h-munakata.github.io/Language-based-Audio-Moment-Retrieval/>
- Released AMR features and evaluation path: <https://doi.org/10.5281/zenodo.13806234>

## Integrity rules

- Every released query must have exactly one prediction and one audio file.
- Query text, source index, benchmark id, filename, and duration are checked
  before boundary decoding.
- Audio groups are the bootstrap unit.
- Missing rows, duplicate rows, illegal intervals, or alignment errors fail the
  run.
- Large per-row predictions and logits remain in
  `/root/autodl-tmp/SpotSound-ICASSP/autoresearch/06_experiments/runs/official_protocol_rerun_20260907`.
