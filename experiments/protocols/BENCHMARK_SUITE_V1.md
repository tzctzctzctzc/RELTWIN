# NOVA benchmark-suite completion protocol

Run date: 2026-09-06. Frozen method parent: `cf57a33`.

This branch adds benchmark adapters and records results without changing the
frozen NOVA model, boundary model, confidence margin, or adaptive radius:

`r_i = min(0.25 seconds, 0.625 * incumbent_event_length_i)`.

## Evaluation tracks

- AEGBench public v3: 3,424 audio items expanded to 9,924 positive queries.
  This is the current public Hugging Face revision
  `49a1d919b6df6717c4a34ef9c01e75aa4b3fc8a5`.  It is not silently equated
  with the paper's stated 3,427-item / 9,790-query table.
- UnAV100-subset public AMR test: 77 audio files and 100 queries from the
  Lighthouse/Zenodo release.  It is not silently equated with SpotSound's
  stated 492-audio / 997-query evaluation, whose manifest is not public.
- TUT Sound Events 2017 public AMR test: 104 queries over the author-released
  60-second WAV segmentation.
- DESED public evaluation: query rows are constructed by grouping the official
  strong labels by `(audio file, event label)` and retaining all disjoint
  intervals for that query.

Every benchmark first runs the locked SpotSound-A adapter to create the
incumbent, then exports the frozen SpanTool evidence and applies the locked
NOVA adaptive-radius decoder.  Metrics are paired on identical rows.  All
manifests and outputs must pass duplicate, missing-row, duration, and audio-file
checks before scores are reported.
