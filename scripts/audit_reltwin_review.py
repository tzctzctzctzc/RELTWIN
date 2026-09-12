#!/usr/bin/env python3
"""Post-hoc aligned relation diagnostics and dependency-aware paired bootstrap."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_reltwin_probe import set_iou as temporal_set_iou


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_id(filename):
    parts = Path(filename).stem.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected ESC-50 source filename: {filename}")
    return parts[1]  # Freesound recording identity, not excerpt/take identity.


def source_components(manifest):
    audio_sources = {}
    for row in manifest:
        audio = Path(row["audio_path"]).name
        sources = tuple(sorted(source_id(f) for f in row["source_files"]))
        if audio in audio_sources and audio_sources[audio] != sources:
            raise ValueError(f"Inconsistent source metadata: {audio}")
        audio_sources[audio] = sources
    parent = {a: a for a in audio_sources}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for audio, sources in sorted(audio_sources.items()):
        for source in sources:
            if source in owner:
                parent[root(audio)] = root(owner[source])
            owner[source] = audio
    members = defaultdict(list)
    for audio in audio_sources:
        members[root(audio)].append(audio)
    stable = {a: min(group) for group in members.values() for a in group}
    return stable, sorted((len(group) for group in members.values()), reverse=True)


def pair_structure(manifest):
    pairs = defaultdict(dict)
    for index, row in enumerate(manifest):
        key = (row["pair_id"], row["template"], row["variant"])
        if row["relation"] in pairs[key]:
            raise ValueError(f"Duplicate relation: {key}")
        pairs[key][row["relation"]] = index
    result = []
    for key, pair in sorted(pairs.items()):
        if set(pair) != {"AB", "BA"}:
            raise ValueError(f"Incomplete relation pair: {key}")
        a, b = pair["AB"], pair["BA"]
        if manifest[a]["audio_path"] != manifest[b]["audio_path"]:
            raise ValueError("Inverse queries must share an audio")
        result.append((a, b))
    return result


def score_predictions(manifest, predictions):
    if len(manifest) != len(predictions):
        raise ValueError("Prediction count mismatch")
    correct = np.asarray([temporal_set_iou(m["annotations"], p) for m, p in zip(manifest, predictions)])
    pairs = pair_structure(manifest)
    pair_acc, swap, joint = [], [], []
    for a, b in pairs:
        wrong_a = temporal_set_iou(manifest[b]["annotations"], predictions[a])
        wrong_b = temporal_set_iou(manifest[a]["annotations"], predictions[b])
        success = bool(correct[a] >= .5 and correct[b] >= .5)
        error = bool(correct[a] <= wrong_a or correct[b] <= wrong_b)
        pair_acc.append(success)
        swap.append(error)
        joint.append(success and not error)
    return {"mIoU": correct, "PairAcc@0.5": np.asarray(pair_acc, float),
            "SwapError": np.asarray(swap, float), "JointPairAcc@0.5": np.asarray(joint, float)}


def aligned_analysis_predictions(path, manifest):
    report = load(path)
    rows = report["query_rows"]
    if len(rows) != len(manifest) or {r["index"] for r in rows} != set(range(len(manifest))):
        raise ValueError(f"Analysis alignment failure: {path}")
    rows = sorted(rows, key=lambda r: r["index"])
    for row, item in zip(rows, manifest):
        for old, new in (("query", "caption"), ("pair_id", "pair_id"), ("template", "template"), ("relation", "relation")):
            if row[old] != item[new]:
                raise ValueError(f"Analysis query mismatch: {path}")
        if abs(temporal_set_iou(item["annotations"], row["prediction"])-row["iou_correct"]) > 1e-9:
            raise ValueError(f"Analysis IoU mismatch: {path}")
    return [r["prediction"] for r in rows]


def cluster_bootstrap(delta_by_seed, groups, repetitions=20000, seed=20260912):
    """Fixed-seed mean effect, shared cluster draws for all paired models/seeds."""
    delta = np.asarray(delta_by_seed, float)
    if delta.ndim != 2 or delta.shape[1] != len(groups):
        raise ValueError("Seed-by-item matrix must align with group labels")
    labels = sorted(set(groups))
    average = delta.mean(axis=0)
    if len(labels) < 2:
        return {"mean_delta_points": float(100*average.mean()), "groups": len(labels), "ci95_points": None}
    indices = {g: np.asarray([i for i, label in enumerate(groups) if label == g]) for g in labels}
    sums = np.asarray([average[indices[g]].sum() for g in labels])
    counts = np.asarray([len(indices[g]) for g in labels])
    draws = np.random.default_rng(seed).integers(0, len(labels), size=(repetitions, len(labels)))
    values = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {"mean_delta_points": float(100*average.mean()), "per_seed_delta_points": (100*delta.mean(axis=1)).tolist(),
            "groups": len(labels), "ci95_points": (100*np.quantile(values, [.025, .975])).tolist(),
            "bootstrap_nonpositive_fraction": float((values <= 0).mean()), "repetitions": repetitions,
            "scope": "Conditional on observed seeds; resampling uncertainty is across clusters, not a seed-population guarantee."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    repo, output = args.repository.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, train = load(args.manifest), load(args.train_manifest)
    if len(manifest) != 320:
        raise ValueError("Expected the frozen 320-query development split")
    def classes(rows): return {r[k] for r in rows for k in ("event_a", "event_b")}
    def sources(rows): return {source_id(f) for r in rows for f in r["source_files"]}
    if classes(train) & classes(manifest) or sources(train) & sources(manifest):
        raise ValueError("Adaptation/development class or source recording overlap")
    query_audio = [Path(r["audio_path"]).name for r in manifest]
    pair_audio = [query_audio[a] for a, _ in pair_structure(manifest)]
    components, sizes = source_components(manifest)
    historical = {
        "sft_seed0": "results/controlled/sft_seed0/analysis.json",
        "rbee_seed0": "results/rbee_seed_0/analysis.json",
        "setpo_seed0": "results/e001/seed_0/analysis.json",
        **{f"{stage}_seed{s}": f"results/e002/seed_{s}/{stage}/reltwin_analysis.json" for s in (1, 2) for stage in ("rbee", "setpo")},
    }
    arrays, summaries, hashes = {}, {}, {}

    def register(name, predictions, origin, path):
        value = score_predictions(manifest, predictions)
        arrays[name] = value
        summaries[name] = {"metrics_percent": {k: float(100*v.mean()) for k, v in value.items()},
                           "threshold_pass_but_swap_error_pairs": int(((value["PairAcc@0.5"] == 1)&(value["SwapError"] == 1)).sum()),
                           "origin": origin}
        hashes[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    for name, relative in historical.items():
        path = repo / relative
        register("cached_"+name, aligned_analysis_predictions(path, manifest), "historical analysis, aligned and recomputed", path)
        original = load(path)
        for new, old in (("mIoU", "query_mIoU"), ("PairAcc@0.5", "pair_acc_0.5"), ("SwapError", "swap_error_rate")):
            if abs(summaries["cached_"+name]["metrics_percent"][new]-100*original[old]) > 1e-9:
                raise ValueError(f"Historical metric regression: {path}, {new}")
    feature_path = repo / "results/e003_probe/reltwin_three_candidates_features.jsonl"
    features = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(features) != 320 or [f["index"] for f in features] != list(range(320)):
        raise ValueError("Official feature cache alignment failure")
    sft_predictions = aligned_analysis_predictions(repo / historical["sft_seed0"], manifest)
    for f, m, sft in zip(features, manifest, sft_predictions):
        if f["query"] != m["caption"] or f["ground_truth"] != m["annotations"] or f["candidates"]["sft"]["prediction"] != sft:
            raise ValueError("Historical official/SFT feature cache mismatch")
    register("cached_official", [f["candidates"]["official"]["prediction"] for f in features],
             "E003 candidate cache labelled official; checkpoint provenance not independently re-established here", feature_path)
    if args.runs_dir:
        for root in sorted(args.runs_dir.iterdir()):
            path = root / "relation_predictions.jsonl"
            if not path.is_file() or root.name.startswith("smoke") or not (root / "relation_summary.json").exists():
                continue
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 320 or {r["index"] for r in rows} != set(range(320)):
                raise ValueError(f"New run alignment failure: {path}")
            rows.sort(key=lambda r: r["index"])
            for row, item in zip(rows, manifest):
                if row["query"] != item["caption"] or row["ground_truth"] != item["annotations"] or row["audio"] != Path(item["audio_path"]).name:
                    raise ValueError(f"New run query mismatch: {path}")
            register("current_"+root.name, [r["prediction"] for r in rows], "new-server shared evaluator", path)
    comparisons = {}

    def compare(name, names, planned):
        available = [(a, b) for a, b in names if a in arrays and b in arrays]
        if not available:
            return
        statistics = {}
        for metric in arrays[available[0][0]]:
            audio = query_audio if metric == "mIoU" else pair_audio
            delta = [arrays[b][metric]-arrays[a][metric] for a, b in available]
            statistics[metric] = {
                "audio_cluster": cluster_bootstrap(delta, audio, args.bootstrap),
                "source_connected_cluster": cluster_bootstrap(delta, [components[a] for a in audio], args.bootstrap),
            }
        comparisons[name] = {"paired_models": available, "planned_seeds_complete": len(available) == planned,
                             "metrics": statistics, "delta_direction": "target minus source; lower SwapError is better"}

    compare("cached_RBEE_minus_SFT_seed0", [("cached_sft_seed0", "cached_rbee_seed0")], 1)
    compare("cached_SetPO_minus_RBEE", [(f"cached_rbee_seed{s}", f"cached_setpo_seed{s}") for s in range(3)], 3)
    compare("current_exchange_increment", [(f"current_no_exchange_seed{s}", f"current_rbee_seed{s}") for s in range(3)], 3)
    compare("current_SetPO_minus_equal_steps", [(f"current_continue_rbee_seed{s}", f"current_setpo_seed{s}") for s in range(3)], 3)
    report = {"data_role": "Previously used development data; post-hoc audit, not an independent test.",
              "queries": 320, "pairs": len(pair_audio), "audio_groups": len(set(query_audio)),
              "source_connected_groups": len(sizes), "source_component_audio_sizes": sizes,
              "train_classes": len(classes(train)), "development_classes": len(classes(manifest)),
              "train_development_class_overlap": [], "train_development_source_recording_overlap": [],
              "seed_resampling": "Not resampled. Same sampled clusters shared by models and all fixed seeds.",
              "iou_implementation": "Historical analyze_reltwin_probe.set_iou retained exactly, including its strict floating-point tie convention.",
              "models": summaries, "comparisons": comparisons, "input_hashes": hashes,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    (output / "relation_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    lines = ["# RelTwin 关系诊断补充审计", "", "既有开发集的事后分析；不是新增独立测试。交换项和继续训练对照须与同服务器评测配对。", "",
             f"320 条查询，160 个关系对，{len(set(query_audio))} 个音频组，{len(sizes)} 个原始来源连通组；最大来源组含 {max(sizes)} 条音频。", "",
             "| 模型 | mIoU | PairAcc@.5 | Swap error | JointPairAcc@.5 |", "|---|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        values = summary["metrics_percent"]
        lines.append("| "+name+" | "+" | ".join(f"{values[k]:.3f}" for k in ("mIoU", "PairAcc@0.5", "SwapError", "JointPairAcc@0.5"))+" |")
    for name, comparison in comparisons.items():
        lines += ["", f"## {name}", "", f"计划种子完整：{comparison['planned_seeds_complete']}。差值为后者减前者，Swap error 越低越好。", "",
                  "| 指标 | 差值（点） | 音频组 95% CI | 来源连通组 95% CI |", "|---|---:|---|---|"]
        for metric, stats in comparison["metrics"].items():
            a, b = stats["audio_cluster"], stats["source_connected_cluster"]
            fmt = lambda v: "不可估计" if v is None else f"[{v[0]:+.3f}, {v[1]:+.3f}]"
            lines.append(f"| {metric} | {a['mean_delta_points']:+.3f} | {fmt(a['ci95_points'])} | {fmt(b['ci95_points'])} |")
    lines += ["", "区间条件于已经观察到的种子；未声称训练种子总体置信保证。按来源连通分组是片段复用依赖的敏感性分析，不消除开发选择偏差。", "", "JointPairAcc 要求一对查询均达到正确窗口 IoU≥0.5，且分别严格偏向各自正确窗口。它是新增的事后诊断指标，不替代原指标。"]
    (output / "RELATION_AUDIT.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"audio_groups": report["audio_groups"], "source_groups": len(sizes), "models": len(summaries), "comparisons": list(comparisons)}))


if __name__ == "__main__":
    main()
