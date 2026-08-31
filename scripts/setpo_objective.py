"""Pure SetPO objectives, separated from model/audio loading for lightweight tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from setpo_candidates import RELATION_EXCHANGE_PERMUTATION


def listwise_cross_entropy(scores, qualities, prediction_temperature, target_temperature):
    target = F.softmax(qualities / target_temperature, dim=0)
    log_prediction = F.log_softmax(scores / prediction_temperature, dim=0)
    loss = -(target * log_prediction).sum()
    return loss, target


def relation_objective(
    scores,
    qualities_ab,
    qualities_ba,
    prediction_temperature,
    target_temperature,
    method_weight,
    exchange_weight,
    reference_ab=None,
    reference_ba=None,
    reference_kl_weight=0.0,
):
    candidate_count = len(RELATION_EXCHANGE_PERMUTATION)
    scores_ab = scores[:candidate_count]
    scores_ba = scores[candidate_count:]
    sft = -(scores_ab[0] + scores_ba[1]) / 2
    listwise_ab, target_ab = listwise_cross_entropy(
        scores_ab, qualities_ab, prediction_temperature, target_temperature
    )
    listwise_ba, target_ba = listwise_cross_entropy(
        scores_ba, qualities_ba, prediction_temperature, target_temperature
    )
    listwise = (listwise_ab + listwise_ba) / 2
    prediction_ab = F.softmax(scores_ab / prediction_temperature, dim=0)
    prediction_ba = F.softmax(scores_ba / prediction_temperature, dim=0)
    permutation = torch.tensor(RELATION_EXCHANGE_PERMUTATION, device=scores.device)
    exchanged_ba = prediction_ba[permutation]
    midpoint = (prediction_ab + exchanged_ba) / 2
    exchange_js = (
        F.kl_div(midpoint.clamp_min(1e-8).log(), prediction_ab, reduction="sum")
        + F.kl_div(midpoint.clamp_min(1e-8).log(), exchanged_ba, reduction="sum")
    ) / 2
    reference_kl = scores.new_zeros(())
    if reference_ab is not None and reference_ba is not None:
        reference_ab = reference_ab.to(device=scores.device, dtype=prediction_ab.dtype)
        reference_ba = reference_ba.to(device=scores.device, dtype=prediction_ba.dtype)
        reference_kl_ab = (
            reference_ab
            * (reference_ab.clamp_min(1e-8).log() - prediction_ab.clamp_min(1e-8).log())
        ).sum()
        reference_kl_ba = (
            reference_ba
            * (reference_ba.clamp_min(1e-8).log() - prediction_ba.clamp_min(1e-8).log())
        ).sum()
        reference_kl = (reference_kl_ab + reference_kl_ba) / 2
    total = (
        sft
        + method_weight * (listwise + exchange_weight * exchange_js)
        + reference_kl_weight * reference_kl
    )
    diagnostics = {
        "sft": float(sft.detach()),
        "relation_listwise": float(listwise.detach()),
        "exchange_js": float(exchange_js.detach()),
        "reference_kl": float(reference_kl.detach()),
        "p_ab_exact": float(prediction_ab[0].detach()),
        "p_ba_exact": float(prediction_ba[1].detach()),
        "target_ab_exact": float(target_ab[0].detach()),
        "target_ba_exact": float(target_ba[1].detach()),
    }
    return total, diagnostics


def rehearsal_objective(
    scores,
    qualities,
    prediction_temperature,
    target_temperature,
    method_weight,
):
    sft = -scores[0]
    listwise, target = listwise_cross_entropy(
        scores, qualities, prediction_temperature, target_temperature
    )
    total = sft + method_weight * listwise
    prediction = F.softmax(scores / prediction_temperature, dim=0)
    return total, {
        "rehearsal_sft": float(sft.detach()),
        "rehearsal_listwise": float(listwise.detach()),
        "rehearsal_p_exact": float(prediction[0].detach()),
        "rehearsal_target_exact": float(target[0].detach()),
    }


def token_kl_from_logits(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
