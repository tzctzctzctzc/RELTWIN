import numpy as np

from nova import interval_gain, intervene_audio


def test_keep_and_drop_reconstruct_wave_away_from_fades():
    wave = np.ones(16000, dtype=np.float32)
    keep = intervene_audio(wave, [(0.25, 0.75)], "keep", fade_ms=0)
    drop = intervene_audio(wave, [(0.25, 0.75)], "drop", fade_ms=0)
    np.testing.assert_allclose(keep + drop, wave)
    assert keep[:3000].sum() == 0
    assert keep[5000:11000].mean() == 1


def test_overlapping_intervals_form_a_union():
    gain = interval_gain(16000, [(0.1, 0.4), (0.3, 0.6)], fade_ms=0)
    assert gain[2000] == 1
    assert gain[8000] == 1
    assert gain[12000] == 0


def test_invalid_mode_is_rejected():
    try:
        intervene_audio(np.ones(8, dtype=np.float32), [(0.0, 0.1)], "unknown")
    except ValueError as error:
        assert "Unknown intervention mode" in str(error)
    else:
        raise AssertionError("Expected invalid intervention mode to fail")
