import numpy as np

from build_esc50_longneedle import choose_starts


def test_choose_starts_are_in_bounds_and_separated():
    starts = choose_starts(np.random.default_rng(0), 3, 1000, 100, 20)
    assert starts == sorted(starts)
    assert min(starts) >= 20
    assert max(starts) + 100 <= 980
    assert min(b - a for a, b in zip(starts, starts[1:])) >= 120
