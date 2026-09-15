import torch

from timeaudio_reltwin import (
    build_timeaudio_sample,
    format_timeaudio_v3,
    parse_timeaudio_intervals,
    rbee_objective,
    timeaudio_answer,
    timeaudio_question,
)


def test_timeaudio_prompt_and_v3_answer_are_canonical():
    assert timeaudio_question("dog barking").startswith("<Speech><SpeechHere></Speech>")
    answer = timeaudio_answer([[1.04, 2.16], [10.0, 12.0]])
    assert answer == "1.0 - 2.2, 10.0 - 12.0"
    assert format_timeaudio_v3(answer) == (
        "<a_1><f_0> - <a_2><f_2>, <a_1><a_0><f_0> - <a_1><a_2><f_0>"
    )


def test_timeaudio_parser_handles_native_and_spotsound_text():
    assert parse_timeaudio_intervals("1.0 - 2.2, 10.0 - 12.0", 11.0) == [
        (1.0, 2.2),
        (10.0, 11.0),
    ]
    assert parse_timeaudio_intervals("from 3.0s to 4.5s", 10.0) == [(3.0, 4.5)]


def test_build_sample_preserves_audio_tensors_and_masks_answer_prefix():
    features = {
        "spectrogram": torch.zeros(1, 80, 3000),
        "raw_wav": torch.zeros(1, 16000),
        "padding_mask": torch.zeros(1, 16000, dtype=torch.bool),
        "duration": torch.tensor([1.0]),
        "audio_path": "/tmp/a.wav",
    }
    sample = build_timeaudio_sample(features, "dog barking", "0.0 - 1.0")
    assert sample["spectrogram"] is features["spectrogram"]
    assert sample["task"] == ["tag"]
    assert sample["text"] == ["<a_0><f_0> - <a_1><f_0>"]


def test_reltwin_objective_prefers_relation_consistent_candidates():
    good = torch.tensor([3.0, 0.0, 0.0, 3.0])
    bad = torch.tensor([0.0, 3.0, 3.0, 0.0])
    good_loss, good_info = rbee_objective(good)
    bad_loss, bad_info = rbee_objective(bad)
    assert good_loss < bad_loss
    assert good_info["candidate_ce"] < bad_info["candidate_ce"]
    assert good_info["exchange_js"] < 1e-6
