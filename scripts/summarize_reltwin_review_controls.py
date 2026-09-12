#!/usr/bin/env python3
"""Strict paired summaries for the frozen queue; never selects models or parameters."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_reltwin_review import cluster_bootstrap, pair_structure, score_predictions, source_components
from interval_metrics import event_f1_iou


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def evaluator_iou(path):
    """Load the exact two pure evaluator functions without importing the GPU stack."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = {"merge_intervals", "temporal_set_iou"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Frozen evaluator metric functions not found")
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["temporal_set_iou"]


def aligned_rows(rows, manifest, iou_function):
    if len(rows) != len(manifest) or {r["index"] for r in rows} != set(range(len(manifest))):
        raise ValueError("Missing, duplicate, or shifted prediction indices")
    rows = sorted(rows, key=lambda r: r["index"])
    for row, truth in zip(rows, manifest):
        if row.get("status") != "ok" or row["query"] != truth["caption"] or row["ground_truth"] != truth["annotations"]:
            raise ValueError("Prediction status, query, or annotation mismatch")
        if Path(row["audio"]).stem != Path(truth["audio_path"]).stem:
            raise ValueError("Audio identity mismatch")
        duration = row["duration_seconds"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid audio duration")
        for start, end in row["prediction"]:
            if not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= duration:
                raise ValueError("Invalid predicted interval")
        iou = row["iou"]
        if not math.isfinite(iou) or not 0 <= iou <= 1 or abs(iou-iou_function(truth["annotations"], row["prediction"])) > 1e-9:
            raise ValueError("Stored IoU does not reproduce the frozen evaluator")
    return rows


def check_durations(rows, known):
    for row in rows:
        audio, duration = Path(row["audio"]).stem, row["duration_seconds"]
        if audio in known and abs(known[audio]-duration) > 1e-9:
            raise ValueError(f"Duration mismatch between queries or models: {audio}")
        known[audio] = duration


def endpoint_status(comparison, metric="JointPairAcc@0.5"):
    if not comparison["planned_seeds_complete"]:
        return "PENDING_ALL_PLANNED_SEEDS"
    stats = comparison["metrics"][metric]
    if stats["audio_cluster"]["mean_delta_points"] <= 0:
        return "NO_POSITIVE_MEAN_INCREMENT"
    bounds = [v["ci95_points"] for v in stats.values()]
    if all(ci is not None and ci[0] > 0 for ci in bounds):
        return "POSITIVE_DEVELOPMENT_CONTRAST_WITH_CLUSTER_SUPPORT"
    return "POSITIVE_MEAN_WITH_UNCERTAIN_CLUSTER_INTERVAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--relation-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    repo, runs, output = args.repository.resolve(), args.runs_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths = {"relation": args.relation_manifest, "public": args.public_manifest}
    manifests = {k: load(p) for k, p in manifest_paths.items()}
    if len(manifests["relation"]) != 320 or len(manifests["public"]) != 400:
        raise ValueError("Expected complete frozen 320/400 query manifests")
    freeze = load(runs / "freeze.json")
    frozen_hashes = freeze["hashes"]
    for path in manifest_paths.values():
        if digest(path) not in frozen_hashes.values():
            raise ValueError(f"Manifest not in frozen inputs: {path}")
    for name in ("evaluate_checkpoint.py", "analyze_reltwin_probe.py"):
        expected = [v for p, v in frozen_hashes.items() if Path(p).name == name]
        if expected != [digest(repo / "scripts" / name)]:
            raise ValueError(f"Scoring source differs from queue freeze: {name}")
    iou_function = evaluator_iou(repo / "scripts/evaluate_checkpoint.py")
    components, sizes = source_components(manifests["relation"])
    audio_groups = {k: [Path(r["audio_path"]).name for r in rows] for k, rows in manifests.items()}
    pair_audio = [audio_groups["relation"][a] for a, _ in pair_structure(manifests["relation"])]
    arrays, records, model_reports = {}, {}, {}
    hashes = {str(p): digest(p) for p in (*manifest_paths.values(), runs / "freeze.json")}
    known_durations = {"relation": {}, "public": {}}
    for root in sorted(runs.iterdir()):
        if not root.is_dir() or root.name.startswith("smoke"):
            continue
        for split in manifests:
            summary_path = root / f"{split}_summary.json"
            if not summary_path.exists():
                continue  # Do not inspect partial predictions and call them complete.
            path = root / f"{split}_predictions.jsonl"
            rows = aligned_rows([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], manifests[split], iou_function)
            check_durations(rows, known_durations[split])
            summary = load(summary_path)
            if summary["expected_count"] != len(rows) or summary["completed_count"] != len(rows):
                raise ValueError(f"Summary counts do not match complete rows: {path}")
            ious = np.asarray([r["iou"] for r in rows])
            for key, value in {"mIoU": ious.mean(), "R1@0.3": (ious >= .3).mean(), "R1@0.5": (ious >= .5).mean()}.items():
                if abs(summary[key]-100*value) > 1e-9:
                    raise ValueError(f"Summary metric regression: {path}, {key}")
            if split == "relation":
                metrics = score_predictions(manifests[split], [r["prediction"] for r in rows])
            else:
                metrics = {"mIoU": ious, **{f"R1@{t}": (ious >= t).astype(float) for t in (.3, .5, .7)},
                           "query_macro_event_F1@0.5": np.asarray([event_f1_iou(r["ground_truth"], r["prediction"])["f1"] for r in rows])}
            arrays[(root.name, split)], records[(root.name, split)] = metrics, rows
            model_reports.setdefault(root.name, {})[split] = {k: float(100*v.mean()) for k, v in metrics.items()}
            hashes[str(path)], hashes[str(summary_path)] = digest(path), digest(summary_path)
            adapter_path = Path(summary["adapter"]) / "adapter_model.safetensors"
            if adapter_path.is_file():
                hashes[str(adapter_path)] = digest(adapter_path)
        train = root / "train_summary.json"
        if train.exists():
            hashes[str(train)] = digest(train)
            model_reports.setdefault(root.name, {})["train"] = load(train)
    contrasts = {
        "exchange_increment": ([(f"no_exchange_seed{s}", f"rbee_seed{s}") for s in range(3)], 3),
        "SetPO_minus_equal_updates": ([(f"continue_rbee_seed{s}", f"setpo_seed{s}") for s in range(3)], 3),
        "RBEE_minus_SFT_seed0": ([("sft_seed0", "rbee_seed0")], 1),
        "RBEE_minus_official": ([("official", f"rbee_seed{s}") for s in range(3)], 3),
        "SetPO_minus_RBEE": ([(f"rbee_seed{s}", f"setpo_seed{s}") for s in range(3)], 3),
        "continuation_minus_RBEE": ([(f"rbee_seed{s}", f"continue_rbee_seed{s}") for s in range(3)], 3),
    }
    comparisons, failures = {}, []
    for name, (pairs, planned) in contrasts.items():
        comparisons[name] = {}
        for split in manifests:
            available = [(a, b) for a, b in pairs if (a, split) in arrays and (b, split) in arrays]
            result = {"paired_models": available, "planned_seeds_complete": len(available) == planned, "metrics": {}}
            for metric in arrays[(available[0][0], split)] if available else ():
                delta = [arrays[(b, split)][metric]-arrays[(a, split)][metric] for a, b in available]
                groups = pair_audio if split == "relation" and metric != "mIoU" else audio_groups[split]
                stats = {"audio_cluster": cluster_bootstrap(delta, groups, args.bootstrap)}
                if split == "relation":
                    stats["source_connected_cluster"] = cluster_bootstrap(delta, [components[a] for a in groups], args.bootstrap)
                result["metrics"][metric] = stats
            result["per_seed_query_outcomes"] = []
            for a, b in available:
                delta = arrays[(b, split)]["mIoU"]-arrays[(a, split)]["mIoU"]
                result["per_seed_query_outcomes"].append({"source": a, "target": b, "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()), "losses": int((delta < -1e-12).sum()), "drops_at_least_0.5": int((delta <= -.5).sum())})
                for index in np.flatnonzero(delta < -1e-12):
                    source, target = records[(a, split)][index], records[(b, split)][index]
                    failures.append({"contrast": name, "split": split, "source_model": a, "target_model": b,
                        "index": int(index), "audio": source["audio"], "query": source["query"], "delta_iou": float(delta[index]),
                        "source_prediction": json.dumps(source["prediction"]), "target_prediction": json.dumps(target["prediction"]),
                        "ground_truth": json.dumps(source["ground_truth"])})
            comparisons[name][split] = result
    decisions = {name: endpoint_status(comparisons[name]["relation"]) for name in ("exchange_increment", "SetPO_minus_equal_updates")}
    state = load(runs / "status.json")
    complete = state["state"] == "COMPLETE" and all(v["planned_seeds_complete"] for c in comparisons.values() for v in c.values())
    report = {"complete": complete, "queue_state": state, "models": model_reports, "comparisons": comparisons,
              "primary_endpoint_decisions": decisions, "input_hashes": hashes, "source_component_sizes": sizes,
              "scope": "Post-hoc development attribution, fixed observed seeds. No natural-relation generalization claim. Historical adapters re-evaluated here; training-runtime equivalence is not established by sharing an evaluator.",
              "decision_scope": "Evidence labels, not model promotion. Pending seeds cannot establish a contrast; all planned seeds run regardless of sign."}
    (output / "review_control_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    columns = ("contrast", "split", "source_model", "target_model", "index", "audio", "query", "delta_iou", "source_prediction", "target_prediction", "ground_truth")
    with (output / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(failures)
    lines = ["# RelTwin 审稿补强对照", "", f"完整队列：{complete}；状态：{state['state']}。缺少的计划种子不得用已有正结果替代。", "",
             "只比较本服务器统一评估流程下的完整 320/400 条预测，不拼接历史总分。来源组区间仅用于关系开发数据。", "",
             "| 模型 | SpotSound mIoU | 关系 mIoU | PairAcc | Swap error | JointPairAcc |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in model_reports.items():
        fields = [value.get("public", {}).get("mIoU"), *[value.get("relation", {}).get(k) for k in ("mIoU", "PairAcc@0.5", "SwapError", "JointPairAcc@0.5")]]
        lines.append("| "+name+" | "+" | ".join("pending" if v is None else f"{v:.6f}" for v in fields)+" |")
    for name in contrasts:
        lines += ["", f"## {name}", "", "| 数据 / 指标 | 完成计划种子 | 差值（点） | 音频组 95% CI | 来源组 95% CI |", "|---|---|---:|---|---|"]
        for split, result in comparisons[name].items():
            for metric, values in result["metrics"].items():
                def fmt(ci): return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                a, s = values["audio_cluster"], values.get("source_connected_cluster", {})
                lines.append(f"| {split} / {metric} | {result['planned_seeds_complete']} | {a['mean_delta_points']:+.3f} | {fmt(a['ci95_points'])} | {fmt(s.get('ci95_points'))} |")
    lines += ["", "## 归因状态", "", *[f"- {k}: {v}" for k, v in decisions.items()], "",
              "全部差值为后者减前者；Swap error 越低越好。区间条件于已观察到的种子，不能消除开发选择偏差。原历史模型与新训练对照共享评估流程，不等于训练硬件／软件完全相同；需要结合默认目标复现实验核查该残余混杂。未做自动提交、默认模型替换或参数选择。"]
    (output / "REVIEW_CONTROL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"complete": complete, "models": len(model_reports), "decisions": decisions}))


if __name__ == "__main__":
    main()
