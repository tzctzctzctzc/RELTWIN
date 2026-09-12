#!/usr/bin/env python3
"""Frozen full-control + runtime-bridge evidence; pure CPU, no model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from audit_reltwin_review import (aligned_analysis_predictions, cluster_bootstrap,
    pair_structure, score_predictions, source_components, source_id, temporal_set_iou)
from summarize_reltwin_review_controls import aligned_rows, check_durations, digest, evaluator_iou, load


def behavior(manifest, rows):
    predictions = [r["prediction"] for r in rows]
    values = score_predictions(manifest, predictions)
    pairs = pair_structure(manifest)
    same = np.asarray([predictions[a] == predictions[b] for a, b in pairs])
    return {
        "same_pairs": int(same.sum()), "same_percent": float(100 * same.mean()),
        "equivalent_pairs": sum(bool(predictions[a] and predictions[b]) and temporal_set_iou(predictions[a], predictions[b]) >= .995 for a, b in pairs),
        "both_empty": sum(not predictions[a] and not predictions[b] for a, b in pairs),
        "joint_correct": int(values["JointPairAcc@0.5"].sum()),
        "pair_pass_without_joint": int((values["PairAcc@0.5"] - values["JointPairAcc@0.5"]).sum()),
    }, same, values["JointPairAcc@0.5"].astype(bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--bridge", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    repo = args.repository.resolve()
    if args.output.exists():
        raise FileExistsError("Evidence outputs are immutable; use a new version")
    control = repo / "results/reltwin_review_controls_20260912"
    final = control / "final_complete"
    original = load(final / "review_summary/review_control_report.json")
    if not original["complete"]:
        raise ValueError("Original controls incomplete")
    manifests = {"relation": load(control / "reltwin_test_manifest.json"), "public": load(final / "public_manifest.json")}
    train = load(control / "reltwin_train_manifest.json")
    test = manifests["relation"]
    assert not {r[k] for r in train for k in ("event_a", "event_b")} & {r[k] for r in test for k in ("event_a", "event_b")}
    assert not {source_id(f) for r in train for f in r["source_files"]} & {source_id(f) for r in test for f in r["source_files"]}
    iou_function = evaluator_iou(repo / "scripts/evaluate_checkpoint.py")
    components, sizes = source_components(test)
    pair_indices = pair_structure(test)
    audio_groups = {s: [Path(r["audio_path"]).name for r in m] for s, m in manifests.items()}
    pair_audio = [audio_groups["relation"][a] for a, b in pair_indices]
    known = {s: {} for s in manifests}
    records, arrays, models, hashes = {}, {}, {}, {}
    roots = {name: final / name for name in original["models"]}
    if args.bridge:
        if load(args.bridge / "status.json")["state"] != "COMPLETE":
            raise ValueError("SFT bridge not complete")
        roots["sft_current_seed0"] = args.bridge / "sft_current_seed0"
        freeze = load(args.bridge / "freeze.json")
        for name in ("evaluate_checkpoint.py", "analyze_reltwin_probe.py"):
            expected = [v for k, v in freeze["hashes"].items() if k.endswith("/scripts/" + name)]
            if expected != [digest(repo / "scripts" / name)]:
                raise ValueError("Bridge evaluator differs")
        hashes[str(args.bridge / "freeze.json")] = digest(args.bridge / "freeze.json")
    for name, root in roots.items():
        models[name] = {}
        for split, manifest in manifests.items():
            path = root / f"{split}_predictions.jsonl"
            sha = digest(path)
            if name != "sft_current_seed0":
                remote = "/root/autodl-tmp/SpotSound-ICASSP/outputs/reltwin_review_controls_20260912/" + name + "/" + path.name
                if sha != original["input_hashes"][remote]:
                    raise ValueError("Control prediction hash mismatch")
            rows = aligned_rows([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], manifest, iou_function)
            check_durations(rows, known[split])
            summary = load(root / f"{split}_summary.json")
            assert summary["expected_count"] == summary["completed_count"] == len(manifest)
            ious = np.asarray([r["iou"] for r in rows])
            values = {"mIoU": ious, **{f"R1@{t}": (ious >= t).astype(float) for t in (.3, .5, .7)}}
            for key in ("mIoU", "R1@0.3", "R1@0.5"):
                assert abs(summary[key] - 100 * values[key].mean()) < 1e-9
            if split == "relation":
                values = score_predictions(manifest, [r["prediction"] for r in rows])
            arrays[name, split], records[name, split] = values, rows
            models[name][split] = {k: float(100 * v.mean()) for k, v in values.items()}
            hashes[str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)] = sha
        if (root / "train_summary.json").exists():
            value = load(root / "train_summary.json")
            models[name]["train"] = {k: v for k, v in value.items() if k != "history"}
        models[name]["behavior"] = behavior(test, records[name, "relation"])[0]
    contrasts = {
        "cand_minus_historical_sft_seed0": [("sft_seed0", "no_exchange_seed0")],
        "exchange_increment": [(f"no_exchange_seed{s}", f"rbee_seed{s}") for s in range(3)],
        "setpo_minus_equal_updates": [(f"continue_rbee_seed{s}", f"setpo_seed{s}") for s in range(3)],
        "cand_minus_official": [("official", f"no_exchange_seed{s}") for s in range(3)],
    }
    if args.bridge:
        contrasts["cand_minus_current_sft_seed0"] = [("sft_current_seed0", "no_exchange_seed0")]
        contrasts["current_minus_historical_sft_seed0"] = [("sft_seed0", "sft_current_seed0")]
    comparisons = {}
    for name, pairs in contrasts.items():
        comparisons[name] = {}
        for split in manifests:
            result = {"paired_models": pairs, "metrics": {}}
            for metric in arrays[pairs[0][0], split]:
                groups = pair_audio if split == "relation" and metric != "mIoU" else audio_groups[split]
                delta = np.asarray([arrays[b, split][metric] - arrays[a, split][metric] for a, b in pairs])
                result["metrics"][metric] = {"audio_cluster": cluster_bootstrap(delta, groups)}
                if split == "relation":
                    result["metrics"][metric]["source_cluster"] = cluster_bootstrap(delta, [components[a] for a in groups])
            delta = arrays[pairs[0][1], split]["mIoU"] - arrays[pairs[0][0], split]["mIoU"]
            result["first_pair_query_outcomes"] = {"wins": int((delta > 1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()), "losses": int((delta < -1e-12).sum()), "catastrophic_drops": int((delta <= -.5).sum())}
            comparisons[name][split] = result
    recovery = {}
    for ref in ("sft_seed0", "sft_current_seed0"):
        if ref not in models:
            continue
        _, mask, ref_joint = behavior(test, records[ref, "relation"])
        recovery[ref] = {"selected_pairs": int(mask.sum()), "audio_groups": len({pair_audio[i] for i in np.flatnonzero(mask)}), "models": {}}
        for name in models:
            _, _, joint = behavior(test, records[name, "relation"])
            recovery[ref]["models"][name] = {"collapsed_pairs_recovered": int((mask & joint).sum()), "joint_recovered_all": int((~ref_joint & joint).sum()), "joint_lost_all": int((ref_joint & ~joint).sum())}
    stage_means = {}
    for stage in ("no_exchange", "rbee", "continue_rbee", "setpo"):
        stage_means[stage] = {split: {metric: {"mean": statistics.mean([models[f"{stage}_seed{s}"][split][metric] for s in range(3)]), "sample_sd": statistics.stdev([models[f"{stage}_seed{s}"][split][metric] for s in range(3)])} for metric in models[f"{stage}_seed0"][split]} for split in ("relation", "public")}
    drift = {}
    historical_paths = {"sft_seed0": "results/controlled/sft_seed0", "rbee_seed0": "results/rbee_seed_0", "setpo_seed0": "results/e001/seed_0", "official": "results/official_spotsound_a"}
    for name, path in historical_paths.items():
        path = repo / path / "public_predictions.jsonl"
        old_rows = aligned_rows([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()], manifests["public"], iou_function)
        check_durations(old_rows, known["public"])
        new = records[name, "public"]
        drift[name] = {"old_mIoU": float(100 * np.mean([r["iou"] for r in old_rows])), "reevaluated_mIoU": models[name]["public"]["mIoU"], "changed_predictions": sum(a["prediction"] != b["prediction"] for a, b in zip(old_rows, new)), "changed_input_token_counts": sum(a["input_tokens"] != b["input_tokens"] for a, b in zip(old_rows, new)), "scope": "Same adapter label/path, not a proof of historical byte identity; no new training in this contrast."}
        hashes[str(path.relative_to(repo))] = digest(path)
    result = {"bridge_complete": args.bridge is not None, "scope": "All relation results are previously observed synthetic development evidence. CIs condition on fixed seeds. No test-set tuning or natural-generalization claim.", "models": models, "stage_means": stage_means, "comparisons": comparisons, "collapse_recovery": recovery, "reevaluation_drift": drift, "audio_groups": {s: len(set(a)) for s, a in audio_groups.items()}, "source_component_sizes": sizes, "train_dev_class_overlap": [], "train_dev_source_overlap": [], "input_hashes": hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bridge_complete": result["bridge_complete"], "models": len(models), "drift": drift, "primary_comparison": comparisons.get("cand_minus_current_sft_seed0", comparisons["cand_minus_historical_sft_seed0"])}))


if __name__ == "__main__":
    main()
