import json

from prepare_public_benchmarks import (
    clotho_audio_name,
    prepare_aegbench,
    prepare_audiogrounding,
    prepare_clotho,
)


def test_prepare_clotho(tmp_path):
    source = tmp_path / "clotho.jsonl"
    source.write_text(
        json.dumps(
            {
                "qid": "1",
                "query": "water splashes",
                "duration": 60,
                "vid": "Wildlife_0.0_60.0",
                "relevant_windows": [[2.5, 7.5]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = prepare_clotho(source)
    assert clotho_audio_name("Wildlife_0.0_60.0") == "Wildlife_00_600.wav"
    assert rows[0]["audio_path"] == "Wildlife_00_600.wav"
    assert rows[0]["annotations"] == [[2.5, 7.5]]


def test_prepare_aegbench_skips_categories_without_ground_truth(tmp_path):
    source = tmp_path / "aeg.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "clip-1",
                    "audio_path": "audio/src/clip.wav",
                    "audio_rel": "audio/src/clip.wav",
                    "duration": 10,
                    "categories": ["bark", "missing"],
                    "clips": [
                        {"category": "bark", "start": 1.0, "end": 2.0},
                        {"category": "bark", "start": 5.0, "end": 6.0},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = prepare_aegbench(source)
    assert len(rows) == 1
    assert rows[0]["caption"] == "bark"
    assert rows[0]["annotations"] == [[1.0, 2.0], [5.0, 6.0]]


def test_prepare_audiogrounding_expands_phrases(tmp_path):
    source = tmp_path / "audiogrounding.json"
    source.write_text(
        json.dumps(
            [
                {
                    "audiocap_id": 42,
                    "audio_id": "Yclip.wav",
                    "tokens": "a dog barks then a person speaks",
                    "phrases": [
                        {"phrase": "a dog barks", "segments": [[0.5, 1.5], [3, 4]]},
                        {"phrase": "a person speaks", "segments": [[5, 7.25]]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = prepare_audiogrounding(source)
    assert [row["caption"] for row in rows] == ["a dog barks", "a person speaks"]
    assert rows[0]["benchmark_id"] == "42:0"
    assert rows[0]["annotations"] == [[0.5, 1.5], [3.0, 4.0]]
