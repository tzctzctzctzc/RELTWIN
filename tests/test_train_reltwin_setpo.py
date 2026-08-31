import torch

from setpo_objective import relation_objective, token_kl_from_logits


def test_reference_kl_is_zero_when_distributions_match():
    scores = torch.tensor(
        [1.0, 0.0, -1.0, 0.5, -0.5, -1.5, 0.8, -0.2, -1.2, 0.3, -0.7, -1.7]
    )
    qualities = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    reference_ab = torch.softmax(scores[:6], dim=0)
    reference_ba = torch.softmax(scores[6:], dim=0)
    base, _ = relation_objective(scores, qualities, qualities, 1.0, 0.15, 0.5, 1.0)
    anchored, diagnostics = relation_objective(
        scores,
        qualities,
        qualities,
        1.0,
        0.15,
        0.5,
        1.0,
        reference_ab,
        reference_ba,
        10.0,
    )
    assert torch.allclose(anchored, base, atol=1e-6)
    assert abs(diagnostics["reference_kl"]) < 1e-6


def test_reference_kl_penalizes_distribution_shift():
    scores = torch.tensor(
        [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]
    )
    qualities = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    uniform = torch.full((6,), 1 / 6)
    base, _ = relation_objective(scores, qualities, qualities, 1.0, 0.15, 0.5, 1.0)
    anchored, diagnostics = relation_objective(
        scores,
        qualities,
        qualities,
        1.0,
        0.15,
        0.5,
        1.0,
        uniform,
        uniform,
        2.0,
    )
    assert diagnostics["reference_kl"] > 0
    assert anchored > base


def test_token_kl_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 1.5]])
    assert abs(float(token_kl_from_logits(logits, logits, 1.0))) < 1e-6


def test_token_kl_penalizes_next_token_shift():
    teacher = torch.tensor([[2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    student = torch.flip(teacher, dims=[-1])
    assert token_kl_from_logits(student, teacher, 1.0) > 0
