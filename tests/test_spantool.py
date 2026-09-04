import itertools
import math
import random

import pytest
import torch

from spantool import (
    SpanToolConfig,
    SpanToolHead,
    bin_spans_to_intervals,
    boundary_targets,
    conditional_segmental_nll,
    intervals_to_bin_spans,
    masked_average_pool,
    perturb_and_map_candidates,
    proposal_span_prior,
    refine_boundaries,
    refine_proposal_intervals,
    segmental_forward_table,
    spantool_loss,
    structured_interval_quality,
    viterbi_decode,
)


def non_overlapping_sets(steps, count):
    spans = [(start, end) for start in range(steps) for end in range(start, steps)]
    for candidate in itertools.combinations(spans, count):
        ordered = sorted(candidate)
        if all(left[1] < right[0] for left, right in zip(ordered[:-1], ordered[1:])):
            yield ordered


def test_masked_average_pool_preserves_partial_final_block():
    values = torch.tensor([[[1.0], [3.0], [5.0]]])
    mask = torch.tensor([[True, True, True]])
    pooled, pooled_mask = masked_average_pool(values, mask, 2)
    assert pooled[:, :, 0].tolist() == [[2.0, 5.0]]
    assert pooled_mask.tolist() == [[True, True]]


def test_segmental_partition_matches_brute_force():
    scores = torch.tensor(
        [[1.0, 0.5, -1.0], [-math.inf, 2.0, 0.25], [-math.inf, -math.inf, 0.75]]
    )
    table = segmental_forward_table(scores, 2)
    brute = []
    for spans in non_overlapping_sets(3, 2):
        brute.append(sum(float(scores[start, end]) for start, end in spans))
    expected = torch.logsumexp(torch.tensor(brute), dim=0)
    assert torch.allclose(table[2][-1], expected)


def test_viterbi_returns_exact_non_overlapping_count():
    scores = torch.full((4, 4), -torch.inf)
    scores[0, 0] = 4.0
    scores[2, 3] = 5.0
    scores[0, 3] = 8.0
    spans, score = viterbi_decode(scores, 2)
    assert spans == [(0, 0), (2, 3)]
    assert score == pytest.approx(9.0)


def test_conditional_nll_prefers_gold_set():
    scores = torch.full((3, 3), -10.0, requires_grad=True)
    with torch.no_grad():
        scores[0, 0] = 4.0
        scores[2, 2] = 4.0
        scores[0, 2] = 1.0
    loss = conditional_segmental_nll(scores, [(0, 0), (2, 2)])
    assert 0 <= float(loss.detach()) < 0.1
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_interval_bin_round_trip_is_bounded():
    spans = intervals_to_bin_spans([[0.11, 0.39], [0.7, 0.95]], 1.0, 10)
    assert spans == [(1, 3), (6, 9)]
    intervals = bin_spans_to_intervals(spans, 1.0, 10)
    assert intervals[0] == pytest.approx((0.1, 0.4))
    assert intervals[1] == pytest.approx((0.6, 1.0))


def test_boundary_targets_align_with_decoder_edges():
    onset, offset = boundary_targets(
        [[1.0, 3.0]], 4.0, 4, sigma_bins=0.5,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert int(onset.argmax()) == 1
    assert int(offset.argmax()) == 2


def test_boundary_refinement_cannot_reorder_close_spans():
    onsets = torch.zeros(30)
    offsets = torch.zeros(30)
    onsets[0] = onsets[29] = 10
    offsets[0] = offsets[29] = 10
    refined = refine_boundaries([(2, 2), (3, 3)], onsets, offsets, 5, 20)
    assert refined[0][1] < refined[1][0]


def test_boundary_refinement_partitions_a_wide_interspan_gap():
    onsets = torch.zeros(8)
    offsets = torch.zeros(8)
    onsets[3] = 10
    offsets[4] = 10
    refined = refine_boundaries([(2, 2), (5, 5)], onsets, offsets, 1, 10)
    assert refined[0][1] < refined[1][0]


def test_proposal_refinement_preserves_cardinality_and_order():
    onsets = torch.zeros(20)
    offsets = torch.zeros(20)
    onsets[3] = onsets[12] = 5
    offsets[7] = offsets[16] = 5
    refined = refine_proposal_intervals(
        [[0.2, 0.9], [1.1, 1.8]], onsets, offsets, 2.0,
        radius_seconds=0.4,
    )
    assert len(refined) == 2
    assert refined[0][1] <= refined[1][0]


def test_proposal_prior_prefers_matching_span():
    prior = proposal_span_prior(
        [[0.2, 0.5]], 1.0, 10,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert prior[2, 4] > prior[0, 9]


def test_structured_quality_penalizes_duplicate_prediction():
    exact = structured_interval_quality([[1, 2]], [[1, 2]], 4)
    duplicate = structured_interval_quality([[1, 2]], [[1, 2], [2.1, 2.5]], 4)
    assert exact == pytest.approx(1.0)
    assert duplicate < exact


def test_head_and_full_loss_are_finite_and_differentiable():
    head = SpanToolHead(
        SpanToolConfig(
            input_dim=12,
            hidden_dim=16,
            transformer_layers=1,
            attention_heads=4,
            primary_pool=2,
            coarse_pool=2,
            max_events=3,
        )
    )
    states = torch.randn(2, 11, 12)
    mask = torch.tensor(
        [[1] * 11, [1] * 8 + [0] * 3], dtype=torch.bool
    )
    durations = torch.tensor([2.2, 1.6])
    output = head(states, mask, durations)
    loss, diagnostics = spantool_loss(
        head,
        output,
        [[[0.2, 0.7], [1.4, 1.8]], [[0.4, 1.2]]],
        durations,
    )
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in diagnostics.values())
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_head_learns_to_mix_multiple_hidden_layers():
    head = SpanToolHead(
        SpanToolConfig(
            input_dim=12,
            layer_count=2,
            hidden_dim=16,
            transformer_layers=1,
            attention_heads=4,
            primary_pool=2,
            coarse_pool=2,
        )
    )
    states = torch.randn(1, 2, 7, 12)
    mask = torch.ones(1, 7, dtype=torch.bool)
    output = head(states, mask, torch.tensor([1.4]))
    output["occupancy_logits"].sum().backward()
    assert head.layer_logits.grad is not None
    assert torch.isfinite(head.layer_logits.grad).all()


def test_perturb_and_map_candidates_are_valid_sets():
    scores = torch.triu(torch.randn(5, 5))
    scores = scores.masked_fill(torch.tril(torch.ones(5, 5), diagonal=-1).bool(), -torch.inf)
    candidates = perturb_and_map_candidates(
        scores, torch.tensor([-2.0, 1.0, 0.5]), 5, 0.25, random.Random(3)
    )
    assert candidates
    for spans in candidates:
        assert all(start <= end for start, end in spans)
        assert all(left[1] < right[0] for left, right in zip(spans[:-1], spans[1:]))
