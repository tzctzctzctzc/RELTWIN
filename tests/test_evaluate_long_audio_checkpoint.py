from long_audio import chunk_bounds, offset_intervals


def test_chunk_bounds_uses_single_window_below_limit():
    assert chunk_bounds(
        1000, 10, chunk_seconds=600, stride_seconds=300
    ) == [(0, 1000)]


def test_chunk_bounds_covers_tail_with_half_overlap():
    assert chunk_bounds(
        16890, 10, chunk_seconds=600, stride_seconds=300
    ) == [
        (0, 6000),
        (3000, 9000),
        (6000, 12000),
        (9000, 15000),
        (10890, 16890),
    ]


def test_offset_intervals_maps_local_prediction_to_global_timeline():
    assert offset_intervals([(12.5, 25.0)], 600.0, 1000.0) == [
        (612.5, 625.0)
    ]


def test_chunk_bounds_rejects_gaps():
    try:
        chunk_bounds(1000, 10, chunk_seconds=30, stride_seconds=31)
    except ValueError as error:
        assert "stride_seconds" in str(error)
    else:
        raise AssertionError("gapped chunk policy was accepted")
