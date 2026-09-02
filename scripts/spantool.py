"""Query-conditioned structured interval decoding for SpotSound.

The language model supplies query-aware audio states.  SpanTool converts those
states into a variable-cardinality set of non-overlapping intervals without
autoregressively generating timestamp text.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Interval = tuple[float, float]
BinSpan = tuple[int, int]
SPANTOOL_PROMPT = (
    "Locate every occurrence of the following sound event in the audio. "
    "The query is: "
)


def timestamped_audio_expansion(
    audio_token: str, num_audio_tokens: int, tokens_per_second: int = 25
) -> str:
    """Expand every audio token, including a final incomplete one-second block."""
    if num_audio_tokens < 0 or tokens_per_second < 1:
        raise ValueError("Invalid audio-token counts")
    chunks = []
    remaining = num_audio_tokens
    timestamp = 0
    while remaining:
        chunk_size = min(tokens_per_second, remaining)
        chunks.append(
            f"timestamp: {timestamp} seconds; feature: " + audio_token * chunk_size
        )
        remaining -= chunk_size
        timestamp += 1
    return "".join(chunks)


@dataclass(frozen=True)
class SpanToolConfig:
    input_dim: int
    layer_count: int = 1
    hidden_dim: int = 256
    transformer_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.1
    primary_pool: int = 5
    coarse_pool: int = 5
    time_features: int = 16
    max_events: int = 8
    max_segment_seconds: float = 0.0
    occupancy_weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def masked_average_pool(
    values: torch.Tensor, mask: torch.Tensor, factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average variable-length sequences in non-overlapping temporal blocks."""
    if factor < 1:
        raise ValueError("Pooling factor must be positive")
    if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError("Expected values [B,T,D] and mask [B,T]")
    if factor == 1:
        return values, mask.bool()

    batch, steps, channels = values.shape
    padded_steps = math.ceil(steps / factor) * factor
    if padded_steps != steps:
        values = F.pad(values, (0, 0, 0, padded_steps - steps))
        mask = F.pad(mask, (0, padded_steps - steps))
    block_mask = mask.reshape(batch, -1, factor).bool()
    weights = block_mask.to(values.dtype).unsqueeze(-1)
    pooled = (values.reshape(batch, -1, factor, channels) * weights).sum(dim=2)
    pooled = pooled / weights.sum(dim=2).clamp_min(1.0)
    return pooled, block_mask.any(dim=2)


def _continuous_time_features(
    durations: torch.Tensor,
    frame_mask: torch.Tensor,
    feature_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if feature_count % 2:
        raise ValueError("time_features must be even")
    batch, steps = frame_mask.shape
    valid_steps = frame_mask.sum(dim=1).clamp_min(1).to(durations.dtype)
    seconds_per_step = durations / valid_steps
    positions = torch.arange(steps, device=frame_mask.device, dtype=durations.dtype) + 0.5
    seconds = positions.unsqueeze(0) * seconds_per_step.unsqueeze(1)
    periods = torch.logspace(
        math.log10(0.08), math.log10(120.0), feature_count // 2,
        device=frame_mask.device, dtype=durations.dtype,
    )
    angles = 2 * math.pi * seconds.unsqueeze(-1) / periods
    features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return (features * frame_mask.unsqueeze(-1)).to(dtype)


class SpanToolHead(nn.Module):
    """A compact multi-scale temporal head over audio-token hidden states."""

    def __init__(self, config: SpanToolConfig):
        super().__init__()
        self.config = config
        if config.layer_count < 1:
            raise ValueError("layer_count must be positive")
        if config.layer_count > 1:
            self.layer_norms = nn.ModuleList(
                nn.LayerNorm(config.input_dim) for _ in range(config.layer_count)
            )
            self.layer_logits = nn.Parameter(torch.zeros(config.layer_count))
        else:
            self.layer_norms = None
            self.register_parameter("layer_logits", None)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.time_projection = nn.Linear(config.time_features, config.hidden_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        self.coarse_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.primary_norm = nn.LayerNorm(config.hidden_dim)
        self.fine_norm = nn.LayerNorm(config.hidden_dim)
        self.occupancy = nn.Linear(config.hidden_dim, 1)
        self.onset = nn.Linear(config.hidden_dim, 1)
        self.offset = nn.Linear(config.hidden_dim, 1)
        self.fine_onset = nn.Linear(config.hidden_dim, 1)
        self.fine_offset = nn.Linear(config.hidden_dim, 1)
        self.coarse_occupancy = nn.Linear(config.hidden_dim, 1)
        self.count_head = nn.Linear(config.hidden_dim, config.max_events + 1)
        self.duration_score = nn.Sequential(
            nn.Linear(2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )
        self.segment_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        frame_states: torch.Tensor,
        frame_mask: torch.Tensor,
        durations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frame_states.ndim == 4:
            if frame_states.shape[1] != self.config.layer_count:
                raise ValueError(
                    f"Expected {self.config.layer_count} hidden layers, "
                    f"received {frame_states.shape[1]}"
                )
            if self.layer_norms is None:
                frame_states = frame_states[:, 0]
            else:
                normalized = torch.stack(
                    [
                        normalizer(frame_states[:, index])
                        for index, normalizer in enumerate(self.layer_norms)
                    ],
                    dim=1,
                )
                weights = F.softmax(self.layer_logits, dim=0).to(normalized.dtype)
                frame_states = (normalized * weights.view(1, -1, 1, 1)).sum(dim=1)
        elif frame_states.ndim != 3 or self.config.layer_count != 1:
            raise ValueError(
                "Expected [B,T,D] for one layer or [B,L,T,D] for layer fusion"
            )
        frame_mask = frame_mask.bool()
        fine = self.input_projection(frame_states)
        fine = fine + self.time_projection(
            _continuous_time_features(
                durations.to(fine.dtype), frame_mask, self.config.time_features, fine.dtype
            )
        )
        fine = fine * frame_mask.unsqueeze(-1)

        primary, primary_mask = masked_average_pool(
            fine, frame_mask, self.config.primary_pool
        )
        primary = self.temporal_encoder(primary, src_key_padding_mask=~primary_mask)
        primary = primary * primary_mask.unsqueeze(-1)

        coarse, coarse_mask = masked_average_pool(
            primary, primary_mask, self.config.coarse_pool
        )
        coarse = self.coarse_projection(coarse) * coarse_mask.unsqueeze(-1)
        coarse_up = torch.repeat_interleave(
            coarse, self.config.coarse_pool, dim=1
        )[:, : primary.shape[1]]
        primary = self.primary_norm(primary + coarse_up) * primary_mask.unsqueeze(-1)

        primary_up = torch.repeat_interleave(
            primary, self.config.primary_pool, dim=1
        )[:, : fine.shape[1]]
        fine = self.fine_norm(fine + primary_up) * frame_mask.unsqueeze(-1)

        weights = primary_mask.to(primary.dtype).unsqueeze(-1)
        pooled = (primary * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "occupancy_logits": self.occupancy(primary).squeeze(-1),
            "onset_logits": self.onset(primary).squeeze(-1),
            "offset_logits": self.offset(primary).squeeze(-1),
            "fine_onset_logits": self.fine_onset(fine).squeeze(-1),
            "fine_offset_logits": self.fine_offset(fine).squeeze(-1),
            "coarse_occupancy_logits": self.coarse_occupancy(coarse).squeeze(-1),
            "count_logits": self.count_head(pooled),
            "frame_mask": frame_mask,
            "primary_mask": primary_mask,
            "coarse_mask": coarse_mask,
        }

    def segment_scores(
        self,
        onset_logits: torch.Tensor,
        offset_logits: torch.Tensor,
        occupancy_logits: torch.Tensor,
        step_seconds: float,
    ) -> torch.Tensor:
        """Build scores for every inclusive [start,end] pair."""
        steps = int(occupancy_logits.numel())
        if steps == 0:
            raise ValueError("Cannot score an empty sequence")
        starts = torch.arange(steps, device=occupancy_logits.device)
        ends = torch.arange(steps, device=occupancy_logits.device)
        lengths = ends.unsqueeze(0) - starts.unsqueeze(1) + 1
        prefix = F.pad(occupancy_logits, (1, 0)).cumsum(dim=0)
        occupancy_sum = prefix[ends.unsqueeze(0) + 1] - prefix[starts.unsqueeze(1)]
        occupancy_mean = occupancy_sum / lengths.clamp_min(1)
        # Invalid lower-triangular entries have non-positive lengths.  Clamp
        # before log1p; masking NaNs afterwards still contaminates gradients.
        safe_lengths = lengths.clamp_min(1).to(occupancy_logits.dtype)
        normalized = safe_lengths / max(steps, 1)
        duration_seconds = safe_lengths * float(step_seconds)
        duration_features = torch.stack(
            (normalized, torch.log1p(duration_seconds) / math.log(121.0)), dim=-1
        )
        duration_score = self.duration_score(duration_features).squeeze(-1)
        scores = (
            onset_logits.unsqueeze(1)
            + offset_logits.unsqueeze(0)
            + self.config.occupancy_weight * occupancy_mean
            + duration_score
            + self.segment_bias
        )
        valid = lengths >= 1
        if self.config.max_segment_seconds > 0:
            valid &= duration_seconds <= self.config.max_segment_seconds + 1e-6
        return scores.masked_fill(~valid, -torch.inf)


def segmental_forward_table(
    span_scores: torch.Tensor, max_events: int
) -> list[torch.Tensor]:
    """Log-sum-exp DP tables for exactly k non-overlapping segments."""
    if span_scores.ndim != 2 or span_scores.shape[0] != span_scores.shape[1]:
        raise ValueError("span_scores must have shape [T,T]")
    steps = span_scores.shape[0]
    rows = [span_scores.new_zeros(steps + 1)]
    for _ in range(max_events):
        used = len(rows)
        previous = rows[-1]
        current = [span_scores.new_tensor(-torch.inf)]
        for end_exclusive in range(1, steps + 1):
            if end_exclusive < used:
                current.append(span_scores.new_tensor(-torch.inf))
                continue
            skip = current[-1]
            add = torch.logsumexp(
                previous[:end_exclusive] + span_scores[:end_exclusive, end_exclusive - 1],
                dim=0,
            )
            current.append(torch.logaddexp(skip, add))
        rows.append(torch.stack(current))
    return rows


def score_bin_set(span_scores: torch.Tensor, spans: Sequence[BinSpan]) -> torch.Tensor:
    if not spans:
        return span_scores.new_zeros(())
    return torch.stack([span_scores[start, end] for start, end in spans]).sum()


def conditional_segmental_nll(
    span_scores: torch.Tensor, target_spans: Sequence[BinSpan]
) -> torch.Tensor:
    count = len(target_spans)
    if count == 0:
        return span_scores.sum() * 0.0
    table = segmental_forward_table(span_scores, count)
    return table[count][-1] - score_bin_set(span_scores, target_spans)


def viterbi_decode(
    span_scores: torch.Tensor, count: int
) -> tuple[list[BinSpan], float]:
    """Return the best set containing exactly ``count`` intervals."""
    if count < 0:
        raise ValueError("count must be non-negative")
    scores = span_scores.detach().float().cpu().numpy()
    steps = scores.shape[0]
    if count == 0:
        return [], 0.0
    negative = float("-inf")
    dp = np.full((count + 1, steps + 1), negative, dtype=np.float64)
    back = np.full((count + 1, steps + 1), -2, dtype=np.int32)
    dp[0, :] = 0.0
    for used in range(1, count + 1):
        for end_exclusive in range(1, steps + 1):
            skip_score = dp[used, end_exclusive - 1]
            candidates = (
                dp[used - 1, :end_exclusive]
                + scores[:end_exclusive, end_exclusive - 1]
            )
            best_start = int(np.argmax(candidates))
            add_score = float(candidates[best_start])
            if add_score > skip_score:
                dp[used, end_exclusive] = add_score
                back[used, end_exclusive] = best_start
            else:
                dp[used, end_exclusive] = skip_score
                back[used, end_exclusive] = -1
    if not math.isfinite(dp[count, steps]):
        return [], negative

    spans: list[BinSpan] = []
    used, end_exclusive = count, steps
    while used:
        start = int(back[used, end_exclusive])
        if start < 0:
            end_exclusive -= 1
            if end_exclusive <= 0:
                raise RuntimeError("Invalid Viterbi backpointer")
            continue
        spans.append((start, end_exclusive - 1))
        used -= 1
        end_exclusive = start
    spans.reverse()
    return spans, float(dp[count, steps])


def normalize_intervals(intervals: Sequence[Sequence[float]], duration: float) -> list[Interval]:
    cleaned = sorted(
        (max(0.0, float(start)), min(duration, float(end)))
        for start, end in intervals
        if float(end) > float(start)
    )
    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end > start]


def intervals_to_bin_spans(
    intervals: Sequence[Sequence[float]], duration: float, bins: int
) -> list[BinSpan]:
    if bins < 1 or duration <= 0:
        raise ValueError("duration and bins must be positive")
    step = duration / bins
    spans = []
    for start, end in normalize_intervals(intervals, duration):
        first = min(bins - 1, max(0, int(math.floor(start / step))))
        last = min(bins - 1, max(first, int(math.ceil(end / step) - 1)))
        if spans and first <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], last))
        else:
            spans.append((first, last))
    return spans


def bin_spans_to_intervals(
    spans: Sequence[BinSpan], duration: float, bins: int
) -> list[Interval]:
    step = duration / bins
    return [(start * step, min(duration, (end + 1) * step)) for start, end in spans]


def rasterize_intervals(
    intervals: Sequence[Sequence[float]], duration: float, bins: int,
    *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    edges = torch.linspace(0.0, duration, bins + 1, device=device, dtype=dtype)
    target = torch.zeros(bins, device=device, dtype=dtype)
    width = duration / bins
    for start, end in normalize_intervals(intervals, duration):
        overlap = (
            torch.minimum(edges[1:], target.new_tensor(end))
            - torch.maximum(edges[:-1], target.new_tensor(start))
        ).clamp_min(0.0)
        target = torch.maximum(target, overlap / width)
    return target


def boundary_targets(
    intervals: Sequence[Sequence[float]], duration: float, bins: int,
    *, sigma_bins: float, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(bins, device=device, dtype=dtype)
    step = duration / bins
    onset = torch.zeros(bins, device=device, dtype=dtype)
    offset = torch.zeros_like(onset)
    for start, end in normalize_intervals(intervals, duration):
        start_bin = start / step
        end_bin = end / step
        # Decoder bin i maps to [i * step, (i + 1) * step].  Supervise the
        # corresponding left and right edges to avoid a half-bin bias.
        onset = torch.maximum(
            onset, torch.exp(-0.5 * ((indices - start_bin) / sigma_bins) ** 2)
        )
        offset = torch.maximum(
            offset, torch.exp(-0.5 * (((indices + 1) - end_bin) / sigma_bins) ** 2)
        )
    return onset, offset


def balanced_binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum().clamp_min(1.0)
    negatives = (1.0 - targets).sum().clamp_min(1.0)
    positive_weight = (negatives / positives).detach().clamp(max=50.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def refine_boundaries(
    spans: Sequence[BinSpan],
    fine_onset_logits: torch.Tensor,
    fine_offset_logits: torch.Tensor,
    primary_pool: int,
    radius_frames: int,
) -> list[BinSpan]:
    """Refine primary-grid boundaries on the original audio-token grid."""
    fine_steps = int(fine_onset_logits.numel())
    refined = []
    for index, (start, end) in enumerate(spans):
        coarse_start = start * primary_pool
        coarse_end = min(fine_steps - 1, (end + 1) * primary_pool - 1)
        # Partition every inter-span gap at its midpoint.  Constraining each
        # side only by the neighbour's original edge leaves overlapping search
        # windows, so two independently selected extrema can still cross.
        previous_limit = 0
        if index:
            previous_end = (spans[index - 1][1] + 1) * primary_pool - 1
            previous_limit = (previous_end + coarse_start) // 2 + 1
        next_limit = fine_steps
        if index + 1 < len(spans):
            next_start = spans[index + 1][0] * primary_pool
            next_limit = (coarse_end + next_start) // 2 + 1
        left = max(previous_limit, coarse_start - radius_frames)
        right = min(coarse_end + 1, coarse_start + radius_frames + 1)
        fine_start = left + int(torch.argmax(fine_onset_logits[left:right]))
        left = max(fine_start, coarse_end - radius_frames)
        right = min(next_limit, coarse_end + radius_frames + 1)
        fine_end = left + int(torch.argmax(fine_offset_logits[left:right]))
        refined.append((fine_start, max(fine_start, fine_end)))

    if any(left[1] >= right[0] for left, right in zip(refined[:-1], refined[1:])):
        raise RuntimeError("Boundary refinement violated interval ordering")
    return refined


def proposal_span_prior(
    proposals: Sequence[Sequence[float]],
    duration: float,
    bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale_bins: float = 5.0,
) -> torch.Tensor:
    """Softly anchor structured decoding to an autoregressive span proposal."""
    targets = intervals_to_bin_spans(proposals, duration, bins)
    if not targets:
        return torch.zeros((bins, bins), device=device, dtype=dtype)
    starts = torch.arange(bins, device=device, dtype=dtype).view(-1, 1)
    ends = torch.arange(bins, device=device, dtype=dtype).view(1, -1)
    priors = [
        -(torch.abs(starts - start) + torch.abs(ends - end)) / max(scale_bins, 1e-6)
        for start, end in targets
    ]
    return torch.stack(priors).amax(dim=0)


def refine_proposal_intervals(
    proposals: Sequence[Sequence[float]],
    fine_onset_logits: torch.Tensor,
    fine_offset_logits: torch.Tensor,
    duration: float,
    *,
    radius_seconds: float = 0.8,
) -> list[Interval]:
    """Preserve proposal cardinality while snapping boundaries to fine logits."""
    fine_steps = int(fine_onset_logits.numel())
    spans = intervals_to_bin_spans(proposals, duration, fine_steps)
    if not spans:
        return []
    radius = max(1, round(radius_seconds / (duration / fine_steps)))
    refined = refine_boundaries(
        spans, fine_onset_logits, fine_offset_logits, 1, radius
    )
    return bin_spans_to_intervals(refined, duration, fine_steps)


def decode_spantool_output(
    head: SpanToolHead,
    output: dict[str, torch.Tensor],
    batch_index: int,
    duration: float,
    *,
    refinement_radius_seconds: float = 0.8,
    proposal_intervals: Sequence[Sequence[float]] | None = None,
    proposal_weight: float = 0.0,
) -> list[Interval]:
    primary_steps = int(output["primary_mask"][batch_index].sum())
    fine_steps = int(output["frame_mask"][batch_index].sum())
    normalized_proposals = (
        normalize_intervals(proposal_intervals, duration)
        if proposal_intervals is not None else None
    )
    count = (
        min(len(normalized_proposals), head.config.max_events)
        if normalized_proposals is not None
        else int(output["count_logits"][batch_index].argmax().clamp(max=head.config.max_events))
    )
    if count == 0:
        return []
    primary_seconds = duration / primary_steps
    scores = head.segment_scores(
        output["onset_logits"][batch_index, :primary_steps],
        output["offset_logits"][batch_index, :primary_steps],
        output["occupancy_logits"][batch_index, :primary_steps],
        primary_seconds,
    )
    if normalized_proposals and proposal_weight > 0:
        scores = scores + proposal_weight * proposal_span_prior(
            normalized_proposals,
            duration,
            primary_steps,
            device=scores.device,
            dtype=scores.dtype,
        )
    spans, _ = viterbi_decode(scores, min(count, primary_steps))
    fine_seconds = duration / fine_steps
    refined = refine_boundaries(
        spans,
        output["fine_onset_logits"][batch_index, :fine_steps],
        output["fine_offset_logits"][batch_index, :fine_steps],
        head.config.primary_pool,
        max(1, round(refinement_radius_seconds / fine_seconds)),
    )
    return bin_spans_to_intervals(refined, duration, fine_steps)


def _interval_iou(left: Interval, right: Interval) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_iou_sum(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    matrix = [[_interval_iou(a, b) for b in right] for a in left]

    @lru_cache(maxsize=None)
    def visit(index: int, used: int) -> float:
        if index == len(left):
            return 0.0
        best = 0.0
        for column in range(len(right)):
            if not used & (1 << column):
                best = max(best, matrix[index][column] + visit(index + 1, used | (1 << column)))
        return best

    return visit(0, 0)


def structured_interval_quality(
    ground_truth: Sequence[Sequence[float]],
    prediction: Sequence[Sequence[float]],
    duration: float,
) -> float:
    """Smooth one-to-one set quality used only by optional structured risk."""
    gt = normalize_intervals(ground_truth, duration)
    pred = normalize_intervals(prediction, duration)
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    matched = _maximum_iou_sum(gt, pred)
    precision = matched / len(pred)
    recall = matched / len(gt)
    soft_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_score = 1.0 - abs(len(gt) - len(pred)) / max(len(gt), len(pred), 1)

    points = sorted({0.0, duration, *(x for interval in gt + pred for x in interval)})
    intersection = union = 0.0
    for start, end in zip(points[:-1], points[1:]):
        middle = (start + end) / 2
        in_gt = any(a <= middle <= b for a, b in gt)
        in_pred = any(a <= middle <= b for a, b in pred)
        intersection += (end - start) * (in_gt and in_pred)
        union += (end - start) * (in_gt or in_pred)
    set_iou = intersection / union if union else 0.0
    return 0.45 * soft_f1 + 0.45 * set_iou + 0.10 * count_score


def perturb_and_map_candidates(
    span_scores: torch.Tensor,
    count_logits: torch.Tensor,
    samples: int,
    noise_scale: float,
    rng: random.Random,
) -> list[list[BinSpan]]:
    """Generate model-dependent interval sets with perturb-and-MAP."""
    candidates: list[list[BinSpan]] = []
    seen = set()
    for sample_index in range(max(1, samples)):
        if sample_index == 0:
            noisy_spans = span_scores
            noisy_count = count_logits
        else:
            seed = rng.randrange(2**31)
            generator = torch.Generator(device=span_scores.device).manual_seed(seed)
            uniform = torch.rand(
                span_scores.shape, generator=generator, device=span_scores.device
            ).clamp_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_spans = span_scores + noise_scale * gumbel
            count_uniform = torch.rand(
                count_logits.shape, generator=generator, device=count_logits.device
            ).clamp_(1e-6, 1 - 1e-6)
            noisy_count = count_logits + noise_scale * (-torch.log(-torch.log(count_uniform)))
        count = int(noisy_count.argmax())
        spans, _ = viterbi_decode(noisy_spans, min(count, span_scores.shape[0]))
        key = tuple(spans)
        if key not in seen:
            seen.add(key)
            candidates.append(spans)
    return candidates


def spantool_loss(
    head: SpanToolHead,
    output: dict[str, torch.Tensor],
    interval_batch: Sequence[Sequence[Sequence[float]]],
    durations: torch.Tensor,
    *,
    structural_weight: float = 1.0,
    occupancy_weight: float = 0.5,
    boundary_weight: float = 0.25,
    count_weight: float = 0.25,
    consistency_weight: float = 0.1,
    risk_weight: float = 0.0,
    risk_samples: int = 0,
    risk_temperature: float = 1.0,
    risk_noise: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the complete structured, frame, scale, and optional risk loss."""
    losses = {name: [] for name in ("structural", "occupancy", "boundary", "count", "consistency", "risk")}
    generator = rng or random.Random(0)
    for index, intervals in enumerate(interval_batch):
        duration = float(durations[index])
        primary_steps = int(output["primary_mask"][index].sum())
        fine_steps = int(output["frame_mask"][index].sum())
        coarse_steps = int(output["coarse_mask"][index].sum())
        if primary_steps == 0 or fine_steps == 0:
            raise ValueError("Every training example must contain audio tokens")

        primary_occ = output["occupancy_logits"][index, :primary_steps]
        primary_on = output["onset_logits"][index, :primary_steps]
        primary_off = output["offset_logits"][index, :primary_steps]
        span_scores = head.segment_scores(
            primary_on, primary_off, primary_occ, duration / primary_steps
        )
        target_spans = intervals_to_bin_spans(intervals, duration, primary_steps)
        losses["structural"].append(conditional_segmental_nll(span_scores, target_spans))

        occupancy_target = rasterize_intervals(
            intervals, duration, primary_steps, device=primary_occ.device, dtype=primary_occ.dtype
        )
        losses["occupancy"].append(balanced_binary_loss(primary_occ, occupancy_target))

        primary_on_target, primary_off_target = boundary_targets(
            intervals, duration, primary_steps, sigma_bins=1.0,
            device=primary_occ.device, dtype=primary_occ.dtype,
        )
        fine_on_target, fine_off_target = boundary_targets(
            intervals, duration, fine_steps, sigma_bins=1.5,
            device=primary_occ.device, dtype=primary_occ.dtype,
        )
        boundary = (
            balanced_binary_loss(primary_on, primary_on_target)
            + balanced_binary_loss(primary_off, primary_off_target)
            + balanced_binary_loss(output["fine_onset_logits"][index, :fine_steps], fine_on_target)
            + balanced_binary_loss(output["fine_offset_logits"][index, :fine_steps], fine_off_target)
        ) / 4
        losses["boundary"].append(boundary)

        target_count = min(len(normalize_intervals(intervals, duration)), head.config.max_events)
        losses["count"].append(
            F.cross_entropy(
                output["count_logits"][index : index + 1],
                torch.tensor([target_count], device=primary_occ.device),
            )
        )

        pooled_probability, pooled_mask = masked_average_pool(
            torch.sigmoid(primary_occ).view(1, -1, 1),
            output["primary_mask"][index : index + 1, :primary_steps],
            head.config.coarse_pool,
        )
        coarse_probability = torch.sigmoid(
            output["coarse_occupancy_logits"][index, :coarse_steps]
        )
        valid = pooled_mask[0, :coarse_steps]
        losses["consistency"].append(
            F.mse_loss(coarse_probability[valid], pooled_probability[0, :coarse_steps, 0][valid])
        )

        if risk_weight > 0 and risk_samples > 0:
            candidates = perturb_and_map_candidates(
                span_scores, output["count_logits"][index], risk_samples, risk_noise, generator
            )
            if tuple(target_spans) not in {tuple(candidate) for candidate in candidates}:
                candidates.append(target_spans)
            candidate_scores = torch.stack(
                [
                    score_bin_set(span_scores, candidate)
                    + F.log_softmax(output["count_logits"][index], dim=0)[
                        min(len(candidate), head.config.max_events)
                    ]
                    for candidate in candidates
                ]
            )
            qualities = candidate_scores.new_tensor(
                [
                    structured_interval_quality(
                        intervals,
                        bin_spans_to_intervals(candidate, duration, primary_steps),
                        duration,
                    )
                    for candidate in candidates
                ]
            )
            probabilities = F.softmax(candidate_scores / risk_temperature, dim=0)
            losses["risk"].append((probabilities * (1.0 - qualities)).sum())
        else:
            losses["risk"].append(primary_occ.sum() * 0.0)

    means = {name: torch.stack(values).mean() for name, values in losses.items()}
    total = (
        structural_weight * means["structural"]
        + occupancy_weight * means["occupancy"]
        + boundary_weight * means["boundary"]
        + count_weight * means["count"]
        + consistency_weight * means["consistency"]
        + risk_weight * means["risk"]
    )
    diagnostics = {name: float(value.detach()) for name, value in means.items()}
    diagnostics["total"] = float(total.detach())
    return total, diagnostics
