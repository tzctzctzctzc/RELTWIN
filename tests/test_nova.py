import numpy as np

from nova import counterfactual_keep_features_fast, interval_gain, intervene_audio


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


def test_neighbor_replacement_preserves_context_and_fills_interval():
    wave = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
    dropped = intervene_audio(
        wave,
        [(0.4, 0.6)],
        "drop",
        fade_ms=0,
        drop_replacement="neighbor",
        context_seconds=0.1,
    )
    np.testing.assert_allclose(dropped[:6000], wave[:6000])
    np.testing.assert_allclose(dropped[10000:], wave[10000:])
    assert np.abs(dropped[7000:9000]).mean() > 0
    assert not np.allclose(dropped[6400:9600], wave[6400:9600])


def test_matched_noise_is_deterministic_and_matches_local_rms():
    wave = np.full(16000, 0.1, dtype=np.float32)
    kwargs = {
        "fade_ms": 0,
        "drop_replacement": "matched_noise",
        "replacement_seed": 17,
        "context_seconds": 0.1,
    }
    first = intervene_audio(wave, [(0.4, 0.6)], "drop", **kwargs)
    second = intervene_audio(wave, [(0.4, 0.6)], "drop", **kwargs)
    np.testing.assert_allclose(first, second)
    fill = first[6400:9600]
    assert abs(float(np.sqrt(np.mean(fill * fill))) - 0.1) < 1e-4
    np.testing.assert_allclose(first[:6400], wave[:6400])


def test_invalid_drop_replacement_is_rejected():
    try:
        intervene_audio(
            np.ones(160, dtype=np.float32),
            [(0.0, 0.01)],
            "drop",
            drop_replacement="unknown",
        )
    except ValueError as error:
        assert "Unknown drop replacement" in str(error)
    else:
        raise AssertionError("Expected invalid drop replacement to fail")


def test_empty_keep_candidate_needs_no_model_forward():
    features = counterfactual_keep_features_fast(
        None,
        None,
        np.zeros(16000, dtype=np.float32),
        "target",
        [],
    )
    assert features == {
        "component_mean": -20.0,
        "component_min": -20.0,
        "duration_fraction": 0.0,
        "interval_count": 0.0,
    }
