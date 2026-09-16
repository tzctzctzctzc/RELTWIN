from pathlib import Path

import numpy as np
import torch

from spotsound_reltwin import (
    capture_rng_state,
    configure_spotsound_parameters,
    group_twin_rows,
    load_bridge_delta,
    restore_rng_state,
    save_bridge_delta,
    set_spotsound_training_mode,
    weighted_schedule,
)


class FakeLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 3, bias=False)
        self.q_proj.lora_A = torch.nn.Linear(3, 2, bias=False)
        self.q_proj.lora_B = torch.nn.Linear(2, 3, bias=False)


class FakeAudioTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.layer_norm = torch.nn.LayerNorm(3)


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLanguage()
        self.multi_modal_projector = torch.nn.Linear(3, 4)
        self.audio_tower = FakeAudioTower()


class FakePeft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = FakeBase()

    def get_base_model(self):
        return self.base_model


def ordinary_rows():
    return [
        {"audio_path": "a.wav", "caption": "first", "annotations": [[0, 1]]},
        {"audio_path": "a.wav", "caption": "second", "annotations": [[2, 3]]},
    ]


def relation_rows():
    return [
        {
            "pair_id": 1,
            "template": "then",
            "variant": 0,
            "relation": "AB",
            "audio_path": "r.wav",
            "caption": "a then b",
            "annotations": [[0, 1]],
        },
        {
            "pair_id": 1,
            "template": "then",
            "variant": 0,
            "relation": "BA",
            "audio_path": "r.wav",
            "caption": "b then a",
            "annotations": [[2, 3]],
        },
    ]


def test_full_scope_trains_only_language_lora_and_audio_bridge():
    model = FakePeft()
    selected, report = configure_spotsound_parameters(model, scope="full")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert len(selected) == report["trainable_tensors"]
    assert report["component_parameters"] == {
        "language_lora": 12,
        "audio_projection": 16,
        "audio_layer_norm": 6,
    }
    assert all(
        "lora_" in name
        or ".multi_modal_projector." in name
        or ".audio_tower.layer_norm." in name
        for name in trainable
    )
    assert not any("audio_tower.encoder" in name for name in trainable)


def test_lora_scope_freezes_the_audio_bridge():
    model = FakePeft()
    _, report = configure_spotsound_parameters(model, scope="lora")
    assert report["component_parameters"] == {"language_lora": 12}


def test_bridge_delta_roundtrip(tmp_path: Path):
    source = FakePeft()
    configure_spotsound_parameters(source, scope="full")
    with torch.no_grad():
        source.base_model.multi_modal_projector.weight.fill_(7)
        source.base_model.audio_tower.layer_norm.weight.fill_(5)
    delta = tmp_path / "bridge.pt"
    save_bridge_delta(source, delta, {"stage": "test"})

    target = FakePeft()
    report = load_bridge_delta(target, delta)
    assert report["loaded_keys"] == 4
    assert report["metadata"] == {"stage": "test"}
    assert torch.all(target.base_model.multi_modal_projector.weight == 7)
    assert torch.all(target.base_model.audio_tower.layer_norm.weight == 5)
    assert not torch.all(target.base_model.audio_tower.encoder.weight == 7)


def test_rng_state_replays_dropout_exactly():
    torch.manual_seed(23)
    dropout = torch.nn.Dropout(0.5).train()
    value = torch.ones(32)
    state = capture_rng_state()
    first = dropout(value)
    restore_rng_state(state)
    second = dropout(value)
    assert torch.equal(first, second)


def test_training_mode_keeps_gradient_checkpointing_eligible():
    model = FakePeft().eval()
    set_spotsound_training_mode(model, scope="full")
    assert model.training
    assert model.base_model.language_model.training
    assert model.base_model.audio_tower.training


def test_weighted_schedule_respects_declared_mix():
    schedule = weighted_schedule([3, 2], [3, 1], 8, np.random.default_rng(0))
    assert sum(source == 0 for source, _ in schedule) == 6
    assert sum(source == 1 for source, _ in schedule) == 2
    assert all(0 <= row < (3 if source == 0 else 2) for source, row in schedule)


def test_twin_grouping_supports_both_training_schemas():
    ordinary, ordinary_schema = group_twin_rows(ordinary_rows())
    relation, relation_schema = group_twin_rows(relation_rows())
    assert ordinary_schema == "ordinary_event_query"
    assert relation_schema == "relation_query"
    assert set(ordinary[0]) == {"AB", "BA"}
    assert set(relation[0]) == {"AB", "BA"}
