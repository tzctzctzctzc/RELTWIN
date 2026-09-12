#!/usr/bin/env python3
"""Strict paired audit of the one frozen timing-cue stress; never tune models."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_reltwin_review import cluster_bootstrap, pair_structure, score_predictions, source_components
from audit_reltwin_paper_evidence import behavior
from summarize_reltwin_review_controls import aligned_rows, check_durations, digest, evaluator_iou, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--stress", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a completed stress audit")
    repo, stress = args.repository.resolve(), args.stress.resolve()
    assert load(stress / "status.json")["state"] == "COMPLETE"
    manifest = load(stress / "manifest.json")
    original = load(repo / "results/reltwin_review_controls_20260912/reltwin_test_manifest.json")
    assert len(manifest) == 160
    assert [r["source_index"] for r in manifest] == [i for i, r in enumerate(original) if r["template"] == "followed_by"]
    freeze = load(stress / "freeze.json")
    assert digest(stress / "manifest.json") in freeze["hashes"].values()
    for name in ("evaluate_checkpoint.py", "spotsound.py"):
        assert [v for k, v in freeze["hashes"].items() if k.endswith("/scripts/" + name)] == [digest(repo / "scripts" / name)]
    iou = evaluator_iou(repo / "scripts/evaluate_checkpoint.py")
    components, sizes = source_components(original)
    known = {"original": {}, "uniform": {}}
    records, arrays, hashes = {}, {}, {}
    for model in ("sft_current_seed0", "no_exchange_seed0"):
        previous = args.bridge / model if model == "sft_current_seed0" else repo / "results/reltwin_review_controls_20260912/final_complete" / model
        old_path = previous / "relation_predictions.jsonl"
        old = aligned_rows([json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines()], original, iou)
        check_durations(old, known["original"])
        for condition, path in (("original", old_path), ("uniform", stress / model / "predictions.jsonl")):
            hashes[str(path)] = digest(path)
            if condition == "original":
                rows = [old[r["source_index"]] for r in manifest]
                labels = [original[r["source_index"]] for r in manifest]
            else:
                rows = aligned_rows([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], manifest, iou)
                assert load(stress / model / "summary.json")["completed_count"] == 160
                labels = manifest
                check_durations(rows, known["uniform"])
                assert all(r["duration_seconds"] == 27 for r in rows)
                for new, old_item in zip(manifest, [original[r["source_index"]] for r in manifest]):
                    assert all(new[k] == old_item[k] for k in ("caption", "relation", "layout", "source_files", "event_a", "event_b"))
            records[model, condition] = rows
            arrays[model, condition] = score_predictions(labels, [r["prediction"] for r in rows])
    pairs = pair_structure(manifest)
    pair_audio = [Path(manifest[a]["audio_path"]).name for a, b in pairs]
    query_audio = [Path(r["audio_path"]).name for r in manifest]
    report = {"complete": True, "scope": "Post-development timing-pattern stress, not natural or independent generalization; original sources and queries reused. No tuning.", "primary_endpoint": "Cand minus matched SFT JointPairAcc on changed BA-first layouts", "subsets": {}, "input_hashes": hashes, "freeze_sha256": digest(stress / "freeze.json"), "source_component_sizes": sizes}
    for subset in ("all", "BA_first", "AB_first"):
        qm = np.asarray([subset == "all" or r["layout"] == subset for r in manifest])
        pm = np.asarray([qm[a] for a, b in pairs])
        value = {"queries": int(qm.sum()), "pairs": int(pm.sum()), "audio_groups": len(set(np.asarray(query_audio)[qm])), "models": {}, "comparisons": {}}
        for condition in ("original", "uniform"):
            for model in ("sft_current_seed0", "no_exchange_seed0"):
                metrics = {k: float(100 * v[qm if k == "mIoU" else pm].mean()) for k, v in arrays[model, condition].items()}
                predictions = [r["prediction"] for r in records[model, condition]]
                metrics["same_answer_pairs"] = sum(predictions[a] == predictions[b] for keep, (a, b) in zip(pm, pairs) if keep)
                value["models"][f"{model}/{condition}"] = metrics
            stats = {}
            for metric in arrays["sft_current_seed0", condition]:
                mask = qm if metric == "mIoU" else pm
                groups = query_audio if metric == "mIoU" else pair_audio
                delta = arrays["no_exchange_seed0", condition][metric][mask] - arrays["sft_current_seed0", condition][metric][mask]
                groups = [g for keep, g in zip(mask, groups) if keep]
                stats[metric] = {"audio_cluster": cluster_bootstrap([delta], groups), "source_cluster": cluster_bootstrap([delta], [components[g] for g in groups])}
            value["comparisons"][f"cand_minus_sft/{condition}"] = stats
        # Identity-layout repeatability is measured, not silently assumed.
        value["unchanged_input_prediction_differences"] = {}
        if subset == "AB_first":
            for model in ("sft_current_seed0", "no_exchange_seed0"):
                value["unchanged_input_prediction_differences"][model] = sum(a["prediction"] != b["prediction"] for keep, a, b in zip(qm, records[model, "original"], records[model, "uniform"]) if keep)
        report["subsets"][subset] = value
    primary = report["subsets"]["BA_first"]["comparisons"]["cand_minus_sft/uniform"]["JointPairAcc@0.5"]["source_cluster"]
    report["primary_result"] = primary
    report["interpretation_label"] = "ADVANTAGE_SURVIVES_THIS_STRESS" if primary["ci95_points"][0] > 0 else "STRESS_ADVANTAGE_NOT_ESTABLISHED"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"primary": primary, "subsets": {s: v["models"] for s, v in report["subsets"].items()}, "identity_repeatability": report["subsets"]["AB_first"]["unchanged_input_prediction_differences"]}))


if __name__ == "__main__":
    main()
