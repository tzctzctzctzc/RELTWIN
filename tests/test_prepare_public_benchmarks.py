import json

from prepare_public_benchmarks import (
    clotho_audio_name,
    prepare_aegbench,
    prepare_amr_jsonl,
    prepare_audiogrounding,
    prepare_clotho,
    prepare_lat_tag,
    parse_lat_timestamp,
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


def test_prepare_amr_jsonl_preserves_multi_interval_sets(tmp_path):
    source = tmp_path / "unav.jsonl"
    source.write_text(
        json.dumps(
            {
                "qid": 7,
                "query": "cars pass",
                "duration": 60,
                "vid": "clip_with.dots",
                "relevant_windows": [[1, 2.5], [10, 15]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = prepare_amr_jsonl(source, benchmark="UnAV100-subset-public100")
    assert rows == [
        {
            "benchmark": "UnAV100-subset-public100",
            "qid": "7",
            "audio_path": "clip_with.dots.wav",
            "caption": "cars pass",
            "annotations": [[1.0, 2.5], [10.0, 15.0]],
            "duration": 60.0,
        }
    ]


def test_prepare_amr_jsonl_rejects_duplicate_qid(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    row = {
        "qid": "same",
        "query": "event",
        "duration": 10,
        "vid": "clip",
        "relevant_windows": [[1, 2]],
    }
    source.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    try:
        prepare_amr_jsonl(source, benchmark="TUT-Sound-Events-2017")
    except ValueError as error:
        assert "Duplicate qid" in str(error)
    else:
        raise AssertionError("duplicate qid was accepted")


def test_prepare_lat_tag_strips_only_fixed_instruction_wrapper(tmp_path):
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text(
        json.dumps({"id": "Bench_EN_1", "duration": 1225.712}) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "tag.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<audio>Please listen to the audio carefully and locate "
                            "the section where a person whistles while a dog barks. "
                            "Please strictly output the result in the format of "
                            "[Start Time - End Time]. Do not output any extra explanatory text."
                        ),
                    },
                    {"role": "assistant", "content": "[11:13 - 12:13]"},
                ],
                "audios": ["Bench_EN_1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = prepare_lat_tag(source, metadata, language="en")
    assert rows == [
        {
            "benchmark": "LAT-Bench-EN-TAG",
            "qid": "Bench_EN_1:673-733",
            "audio_group": "Bench_EN_1",
            "audio_path": "Bench_EN_1.wav",
            "caption": "the section where a person whistles while a dog barks.",
            "annotations": [[673.0, 733.0]],
            "duration": 1225.712,
            "metadata_duration": 1225.712,
            "duration_mismatch": False,
            "released_prompt": (
                "<audio>Please listen to the audio carefully and locate "
                "the section where a person whistles while a dog barks. "
                "Please strictly output the result in the format of "
                "[Start Time - End Time]. Do not output any extra explanatory text."
            ),
        }
    ]


def test_prepare_lat_tag_uses_audio_duration_over_incorrect_metadata(tmp_path):
    import wave

    metadata = tmp_path / "meta.jsonl"
    metadata.write_text(
        json.dumps({"id": "Bench_EN_36", "duration": 6.0}) + "\n",
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    with wave.open(str(audio_dir / "Bench_EN_36.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(10)
        handle.writeframes(b"\x00\x00" * 120)
    source = tmp_path / "tag.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<audio>Please listen to the audio carefully and locate "
                            "the late event. Please strictly output the result in the format of "
                            "[Start Time - End Time]. Do not output any extra explanatory text."
                        ),
                    },
                    {"role": "assistant", "content": "[00:08 - 00:10]"},
                ],
                "audios": ["Bench_EN_36"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = prepare_lat_tag(
        source, metadata, language="en", audio_dir=audio_dir
    )
    assert rows[0]["duration"] == 12.0
    assert rows[0]["metadata_duration"] == 6.0
    assert rows[0]["duration_mismatch"] is True


def test_parse_lat_timestamp_supports_hour_form_and_rejects_invalid():
    assert parse_lat_timestamp("01:02:03") == 3723.0
    try:
        parse_lat_timestamp("12:60")
    except ValueError as error:
        assert "Invalid LAT timestamp" in str(error)
    else:
        raise AssertionError("invalid LAT timestamp was accepted")


def test_prepare_lat_tag_requires_metadata(tmp_path):
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text("", encoding="utf-8")
    source = tmp_path / "tag.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "unsupported"},
                    {"role": "assistant", "content": "[00:01 - 00:02]"},
                ],
                "audios": ["missing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        prepare_lat_tag(source, metadata, language="en")
    except ValueError as error:
        assert "Missing LAT metadata" in str(error)
    else:
        raise AssertionError("LAT record without metadata was accepted")
