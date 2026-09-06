# Adaptive-radius post-hoc development protocol v1

AudioGrounding-v2 and the local AEGBench expansion exposed the same failure of
the frozen Boundary Utility decoder: a fixed 0.25-second trust region is too
large for sub-second events. From this point onward, both external sets are
treated as **post-hoc development data**, not untouched test sets. Scores used
to select the radius policy must not be reported as independent leaderboard
results.

The only changed factor in v1 is the boundary trust region. For incumbent event
length `L_i`, the radius is

```text
r_i = min(0.25 seconds, rho * L_i)
```

One global `rho` is shared by AudioGrounding and AEGBench. The utility model,
bootstrap ensemble, feature definition, decision margin (0.02), incumbent
predictions, and cached logits remain frozen. Candidate `rho` values are tried
in a declared order and every attempt is recorded, including ineffective ones.

A configuration is retained only if both datasets have zero new catastrophic
regressions, neither dataset loses mIoU relative to its incumbent, and the
combined row-weighted mIoU change is positive. Ties prefer the smaller `rho`.
SpotSound and Clotho are reserved for a subsequent no-regression back-test after
the adaptive policy is frozen; they are not used to select `rho`.
