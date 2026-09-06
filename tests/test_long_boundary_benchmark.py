from prepare_long_boundary_benchmark import build_manifest, localize_intervals
from restore_long_boundary_predictions import restore_record


def test_localize_intervals_clips_to_selected_window():
    assert localize_intervals([(590.0, 620.0), (700.0, 720.0)], 600.0, 900.0) == [
        (0.0, 20.0),
        (100.0, 120.0),
    ]


def test_build_manifest_converts_global_incumbent_to_local_window():
    annotations = [
        {
            "audio_group": "clip",
            "audio_path": "clip.wav",
            "caption": "event",
            "annotations": [[673.0, 733.0]],
            "qid": "q0",
        }
    ]
    predictions = [
        {
            "index": 0,
            "audio": "clip.wav",
            "query": "event",
            "prediction": [[672.9, 772.9]],
            "duration_seconds": 871.9,
            "selected_chunk": 1,
            "chunk_bounds_seconds": [[0.0, 600.0], [271.9, 871.9]],
            "chunk_detection_log_odds": [-1.0, 3.0],
        }
    ]
    rows = build_manifest(
        annotations, predictions, benchmark_name="LAT", expected_rows=1
    )
    assert rows[0]["annotations"] == [(401.1, 461.1)]
    assert rows[0]["incumbent_prediction"] == [(401.0, 501.0)]
    assert rows[0]["audio_window_start_seconds"] == 271.9
    assert rows[0]["full_duration"] == 871.9


def test_restore_record_maps_every_decoder_action_to_global_time():
    local = {
        "source_index": 0,
        "audio_window_start_seconds": 271.9,
        "full_duration": 871.9,
        "global_annotations": [[673.0, 733.0]],
        "incumbent_prediction": [[401.0, 501.0]],
        "candidate_prediction": [[401.1, 500.9]],
        "selected_prediction": [[401.1, 500.9]],
        "fixed_threshold_prediction": [[401.0, 501.0]],
        "switch": True,
    }
    restored = restore_record(local)
    assert restored["incumbent_prediction"] == [(672.9, 772.9)]
    assert restored["selected_prediction"] == [(673.0, 772.8)]
    assert restored["ground_truth"] == [(673.0, 733.0)]
    assert restored["delta_iou"] == (
        restored["selected_iou"] - restored["incumbent_iou"]
    )
