#!/usr/bin/env python3
"""Describe query-response failures using frozen predictions; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_reltwin_review import pair_structure, score_predictions, temporal_set_iou
from summarize_reltwin_review_controls import aligned_rows, check_durations, evaluator_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing analysis")
    repo = args.repository.resolve()
    root = repo / "results/reltwin_review_controls_20260912"
    manifest_path = root / "reltwin_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = root / "final_complete"
    report = json.loads((final / "review_summary/review_control_report.json").read_text(encoding="utf-8"))
    assert report["complete"] and len(report["models"]) == 14
    pairs = pair_structure(manifest)
    assert len(manifest) == 320 and len(pairs) == 160
    metric = evaluator_iou(repo / "scripts/evaluate_checkpoint.py")
    records, metrics, masks, hashes, known = {}, {}, {}, {}, {}
    for name in sorted(report["models"]):
        path = final / name / "relation_predictions.jsonl"
        raw = path.read_bytes()
        remote = "/root/autodl-tmp/SpotSound-ICASSP/outputs/reltwin_review_controls_20260912/" + name + "/relation_predictions.jsonl"
        assert hashlib.sha256(raw).hexdigest() == report["input_hashes"][remote]
        rows = aligned_rows([json.loads(line) for line in raw.decode("utf-8").splitlines()], manifest, metric)
        check_durations(rows, known)
        records[name] = rows
        predictions = [r["prediction"] for r in rows]
        values = score_predictions(manifest, predictions)
        for key, array in values.items():
            assert abs(100 * array.mean() - report["models"][name]["relation"][key]) < 1e-9
        metrics[name] = values
        masks[name] = np.asarray([predictions[a] == predictions[b] for a, b in pairs])
        hashes[str(path.relative_to(repo))] = hashlib.sha256(raw).hexdigest()
    reference_mask = masks["sft_seed0"]
    result = {
        "scope": "Post-hoc descriptive reanalysis of existing synthetic development predictions. No training or inference. No claim of independent generalization or causal attribution.",
        "definitions": {
            "exact_same_output": "Two inverse queries have identical serialized interval lists; both-empty is also counted and separately reported.",
            "nonempty_equivalent": "Both nonempty and set-IoU between the two predictions >= 0.995.",
            "recovery": "JointPairAcc@0.5 success on the subset selected by SFT seed 0 exact_same_output. Output change alone does not count as recovery.",
            "comparison_scope": "Only SFT seed 0 exists; other seeds use that same descriptive reference, not matched-SFT causal effects.",
        },
        "queries": len(manifest), "pairs": len(pairs),
        "audio_groups": len({r["audio_path"] for r in manifest}),
        "reference_failure_pairs": int(reference_mask.sum()),
        "reference_failure_audio_groups": len({manifest[a]["audio_path"] for keep, (a, b) in zip(reference_mask, pairs) if keep}),
        "models": {}, "reference_failure_cases": [], "input_hashes": hashes,
    }
    for name, rows in records.items():
        predictions = [r["prediction"] for r in rows]
        joint = metrics[name]["JointPairAcc@0.5"].astype(bool)
        result["models"][name] = {
            "exact_same_output_pairs": int(masks[name].sum()),
            "exact_same_output_percent": float(100 * masks[name].mean()),
            "both_empty_pairs": sum(not predictions[a] and not predictions[b] for a, b in pairs),
            "nonempty_equivalent_pairs": sum(bool(predictions[a] and predictions[b]) and temporal_set_iou(predictions[a], predictions[b]) >= .995 for a, b in pairs),
            "joint_correct_pairs": int(joint.sum()),
            "reference_failure_pairs_joint_recovered": int((reference_mask & joint).sum()),
        }
    for pair_index, (a, b) in enumerate(pairs):
        if not reference_mask[pair_index]:
            continue
        result["reference_failure_cases"].append({
            "query_indices": [a, b], "audio_group": Path(manifest[a]["audio_path"]).name,
            "queries": [manifest[a]["caption"], manifest[b]["caption"]],
            "ground_truth": [manifest[a]["annotations"], manifest[b]["annotations"]],
            "predictions": {name: [rows[a]["prediction"], rows[b]["prediction"]] for name, rows in records.items()},
            "joint_correct": {name: bool(values["JointPairAcc@0.5"][pair_index]) for name, values in metrics.items()},
        })
    result["input_hashes"][str(manifest_path.relative_to(repo))] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("pairs", "audio_groups", "reference_failure_pairs", "reference_failure_audio_groups", "models")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
