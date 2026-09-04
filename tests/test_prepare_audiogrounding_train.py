from prepare_audiogrounding_train import (
    audio_id,
    convert_record,
    repair_unknown_length_flac,
)


def test_audio_id_normalizes_official_youtube_prefix():
    assert audio_id("Yabc123.flac") == "abc123"
    assert audio_id("abc123.wav") == "abc123"


def test_convert_record_groups_repeated_events_by_query():
    record = {
        "events": [
            {"start_time": 3.0, "end_time": 4.0, "caption": "dog bark"},
            {"start_time": 1.0, "end_time": 2.0, "caption": "dog bark"},
            {"start_time": 0.0, "end_time": 0.0, "caption": "bad"},
        ]
    }
    rows = convert_record(record, "Yclip.flac")
    assert len(rows) == 1
    assert rows[0]["caption"] == "dog bark"
    assert rows[0]["annotations"] == [[1.0, 2.0], [3.0, 4.0]]


def test_repair_unknown_length_flac_only_changes_total_samples(tmp_path):
    source = tmp_path / "source.flac"
    destination = tmp_path / "fixed.flac"
    stream_info = (48000 << 44) | (15 << 36)
    source.write_bytes(b"fLaC" + bytes([0, 0, 0, 34]) + b"\0" * 10 + stream_info.to_bytes(8, "big") + b"\0" * 24)
    repair_unknown_length_flac(source, destination, 10.0)
    repaired = destination.read_bytes()
    assert repaired[:18] == source.read_bytes()[:18]
    assert int.from_bytes(repaired[18:26], "big") & ((1 << 36) - 1) == 480000
