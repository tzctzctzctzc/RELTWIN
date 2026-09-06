import pytest

from prepare_boundary_benchmark import build_manifest


def _annotation():
    return {
        "benchmark_id": "item:0",
        "audio_path": "nested/audio.wav",
        "caption": "a dog barks",
        "annotations": [[1.0, 2.0]],
        "duration": 4.0,
        "hardcase_tags": ["RO"],
    }


def _prediction():
    return {
        "index": 0,
        "benchmark_id": "item:0",
        "audio": "audio.wav",
        "query": "a dog barks",
        "prediction": [[0.9, 2.1]],
        "duration_seconds": 4.0,
    }


def test_build_manifest_preserves_alignment_and_incumbent():
    rows = build_manifest(
        [_annotation()],
        [_prediction()],
        benchmark_name="External-Bench",
        incumbent_name="official",
        expected_rows=1,
    )
    assert rows == [
        {
            "benchmark": "External-Bench",
            "source": "External-Bench",
            "source_index": 0,
            "audio_group": "nested/audio.wav",
            "audio_path": "nested/audio.wav",
            "caption": "a dog barks",
            "annotations": [[1.0, 2.0]],
            "duration": 4.0,
            "incumbent_name": "official",
            "incumbent_prediction": [[0.9, 2.1]],
            "benchmark_id": "item:0",
            "hardcase_tags": ["RO"],
        }
    ]


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("index", 1, "indices are not contiguous"),
        ("benchmark_id", "other", "benchmark id mismatch"),
        ("audio", "other.wav", "audio mismatch"),
        ("query", "a cat meows", "query mismatch"),
        ("duration_seconds", 5.0, "duration mismatch"),
    ],
)
def test_build_manifest_rejects_misalignment(field, value, pattern):
    prediction = _prediction()
    prediction[field] = value
    with pytest.raises(ValueError, match=pattern):
        build_manifest(
            [_annotation()],
            [prediction],
            benchmark_name="External-Bench",
            incumbent_name="official",
            expected_rows=1,
        )


def test_build_manifest_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate prediction index"):
        build_manifest(
            [_annotation(), _annotation()],
            [_prediction(), _prediction()],
            benchmark_name="External-Bench",
            incumbent_name="official",
            expected_rows=2,
        )
