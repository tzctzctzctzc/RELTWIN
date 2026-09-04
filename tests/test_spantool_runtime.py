import numpy as np
import torch
import pytest

from spantool import timestamped_audio_expansion
from spantool_runtime import (
    artifact_sha256,
    extract_audio_states,
    prepare_spantool_input,
    select_audio_states,
)


class _Outputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


def test_query_precedes_audio_in_spantool_conversation():
    class Processor:
        @staticmethod
        def apply_chat_template(conversation, **_kwargs):
            return conversation

    conversation = prepare_spantool_input(
        Processor(), np.zeros(160, dtype=np.float32), "dog bark"
    )
    content = conversation[0]["content"]
    assert [item["type"] for item in content] == ["text", "audio"]
    assert "dog bark" in content[0]["text"]


def test_spantool_processor_keeps_partial_second_tokens():
    expanded = timestamped_audio_expansion("<sound>", 27)
    assert expanded.count("<sound>") == 27
    assert "timestamp: 1 seconds" in expanded


def test_extract_audio_states_pads_variable_counts():
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    input_ids = torch.tensor([[9, 1, 1, 2, 2], [1, 2, 1, 2, 1]])
    states, mask = extract_audio_states(hidden, input_ids, 1)
    assert states.shape == (2, 3, 3)
    assert mask.tolist() == [[True, True, False], [True, True, True]]


def test_select_audio_states_stacks_requested_layers():
    ids = torch.tensor([[5, 7, 7]])
    hidden = tuple(torch.full((1, 3, 2), float(index)) for index in range(4))
    states, mask = select_audio_states(_Outputs(hidden), [-3, -1], ids, 7)
    assert states.shape == (1, 2, 2, 2)
    assert states[0, :, 0, 0].tolist() == [1.0, 3.0]
    assert mask.tolist() == [[True, True]]


def test_artifact_hash_is_content_based(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "weights").write_bytes(b"same")
    (right / "weights").write_bytes(b"same")
    assert artifact_sha256(left) == artifact_sha256(right)
    (right / "weights").write_bytes(b"changed")
    assert artifact_sha256(left) != artifact_sha256(right)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        artifact_sha256(tmp_path / "empty")
