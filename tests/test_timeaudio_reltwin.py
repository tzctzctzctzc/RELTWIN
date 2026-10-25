import numpy as np
import torch

from timeaudio_reltwin import (
    build_timeaudio_sample,
    configure_reltwin_parameters,
    format_timeaudio_v3,
    group_twin_rows,
    parse_timeaudio_intervals,
    prepare_beats_waveform,
    rbee_objective,
    timeaudio_answer,
    timeaudio_question,
)


class TinyTimeAudio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llama_model = torch.nn.Module()
        self.llama_model.lora_A = torch.nn.Linear(2, 2, bias=False)
        self.llama_model.base = torch.nn.Linear(2, 2, bias=False)
        self.speech_Qformer = torch.nn.Linear(2, 2)
        self.speech_query_tokens = torch.nn.Parameter(torch.zeros(1, 1, 2))
        self.speech_llama_proj = torch.nn.Linear(2, 2)
        self.abs_time_position_embedding = torch.nn.Parameter(torch.zeros(4, 2))
        self.ln_speech = torch.nn.LayerNorm(2)
        self.ln_audio = torch.nn.LayerNorm(2)
        self.speech_encoder = torch.nn.Linear(2, 2)
        self.beats = torch.nn.Linear(2, 2)


def test_beats_waveform_preserves_official_float64_path():
    waveform = prepare_beats_waveform(np.asarray([0.25, -0.5], dtype=np.float32))
    assert waveform.dtype == torch.float64
    assert waveform.shape == (1, 2)


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


def test_group_twin_rows_accepts_two_ordinary_queries_per_audio():
    rows = [
        {"audio_path": "pair.wav", "caption": "hand saw", "annotations": [[6.0, 10.0]]},
        {"audio_path": "pair.wav", "caption": "crow", "annotations": [[1.0, 5.0]]},
    ]
    groups, schema = group_twin_rows(rows)
    assert schema == "ordinary_event_query"
    assert len(groups) == 1
    assert groups[0]["AB"]["caption"] == "crow"
    assert groups[0]["BA"]["caption"] == "hand saw"


def test_group_twin_rows_preserves_relation_manifest():
    common = {"audio_path": "pair.wav", "pair_id": 3, "template": "then", "variant": 0}
    rows = [
        {**common, "relation": "BA", "caption": "B then A", "annotations": [[6.0, 10.0]]},
        {**common, "relation": "AB", "caption": "A then B", "annotations": [[1.0, 5.0]]},
    ]
    groups, schema = group_twin_rows(rows)
    assert schema == "relation_query"
    assert groups[0]["AB"]["caption"] == "A then B"
    assert groups[0]["BA"]["caption"] == "B then A"


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


def test_full_parameter_scope_updates_temporal_path_but_not_backbones():
    model = TinyTimeAudio()
    trainable, report = configure_reltwin_parameters(model, scope="full")
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "llama_model.lora_A.weight" in names
    assert "speech_Qformer.weight" in names
    assert "speech_query_tokens" in names
    assert "speech_llama_proj.weight" in names
    assert "abs_time_position_embedding" in names
    assert "ln_speech.weight" in names and "ln_audio.weight" in names
    assert "llama_model.base.weight" not in names
    assert "speech_encoder.weight" not in names
    assert "beats.weight" not in names
    assert report["trainable_parameters"] == sum(parameter.numel() for parameter in trainable)
