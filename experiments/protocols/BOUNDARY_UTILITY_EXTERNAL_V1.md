# Boundary Utility external-benchmark protocol v1

This protocol applies the frozen Boundary Utility model released on `main@7b3148b`
to cached official SpotSound-A predictions. It does not refit the utility model,
radius, margin, verifier, or feature backbone on either public test set.

Evaluation order:

1. AudioGrounding-v2, 997 normalized query rows.
2. AEGBench local expansion, 9,924 normalized category-query rows.

The AEGBench result is explicitly labeled `AEGBench-local-expansion-v1`: the local
manifest expands every positive category in each downloaded item and does not match
the item/query counts reported by the later Auto-AEG v4 paper. It is diagnostic and
must not be presented as a directly comparable official-paper score until the
revision and expansion rules are reconciled.

For 114 rows from segmented `yt_life` audio, the downloaded manifest stores the
duration of the original full video rather than the materialized segment. These
rows use the cached inference-time audio duration, after checking that every ground
truth interval lies inside it. The manifest records both durations and a mismatch
flag; AudioGrounding keeps strict duration equality.

The incumbent is the cached official SpotSound-A prediction for each row. Boundary
features use the frozen SetPO seed-1 adapter and frozen SpanTool checkpoint, matching
the released Boundary Utility evaluation. The selected development parameters remain
`alpha=0.1`, `radius_seconds=0.25`, and `margin=0.02`.
