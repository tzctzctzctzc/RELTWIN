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
  The released query `qid=83` has `[26, 48]` as ground truth despite a
  46-second metadata duration and a 46.022-second WAV.  Its released interval
  is preserved with an explicit 2-second overshoot flag rather than silently
  clipped.
- TUT Sound Events 2017 public AMR test: 104 queries over the author-released
  60-second WAV segmentation.
- DESED public evaluation: query rows are constructed by grouping the official
  strong labels by `(audio file, event label)` and retaining all disjoint
  intervals for that query.  The official Zenodo archive MD5 is verified as
  `67697d75672b1b4418a54bc5f3a443e1`; its bundled public annotations exactly
  match the upstream DESED repository file.  This produces 1,112 queries over
  692 WAV files, including 556 queries with multiple target intervals.
- LAT-Bench English TAG: 104 long-form audio files and 426 released queries.
  The fixed prompt wrapper is removed before the semantic query is passed to
  the SpotSound harness, while the full released prompt is retained in each
  normalized row for audit.  Audio Flamingo 3 silently truncates inputs beyond
  600 seconds, so LAT uses deterministic 600-second windows with a 300-second
  stride.  The frozen SpotSound Yes/No existence score selects one window and
  grounding is run only there; local timestamps are then mapped to the global
  timeline.  This selection uses no LAT labels or tuned threshold.
  The public `Bench_EN_97.wav` is 349.693 seconds while metadata says 818
  seconds; two released TAG intervals begin after the WAV ends.  Both rows stay
  in the 426-query primary score as flagged unreachable labels, with a separate
  424-valid-row diagnostic reported rather than silently deleting them.

Every benchmark first runs the locked SpotSound-A adapter to create the
incumbent, then exports the frozen SpanTool evidence and applies the locked
NOVA adaptive-radius decoder.  Metrics are paired on identical rows.  All
manifests and outputs must pass duplicate, missing-row, duration, and audio-file
checks before scores are reported.

For LAT boundary verification, the verifier receives the incumbent span plus
two seconds of context on each side.  This crop is determined without labels
and contains every feasible edit under the frozen 0.25-second radius cap.
Boundary crops longer than 300 seconds are retained as explicit identity
actions with `boundary_crop_exceeds_memory_budget` rather than risking an OOM;
10 of 426 LAT rows meet this condition.
