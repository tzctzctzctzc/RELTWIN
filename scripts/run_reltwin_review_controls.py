#!/usr/bin/env python3
"""Durable single-GPU queue for predeclared RelTwin review controls (Linux)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_predictions(path, manifest, expected=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    truth = read(manifest)
    expected = len(truth) if expected is None else expected
    if len(rows) != expected or {r["index"] for r in rows} != set(range(expected)):
        raise ValueError(f"Missing, duplicate, or shifted indices: {path}")
    for row in rows:
        gt = truth[row["index"]]
        if row["query"] != gt["caption"] or row["ground_truth"] != gt["annotations"]:
            raise ValueError(f"Query or target mismatch: {path}, {row['index']}")
        if Path(row["audio"]).stem != Path(gt["audio_path"]).stem or row["duration_seconds"] <= 0:
            raise ValueError(f"Audio or duration mismatch: {path}, {row['index']}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = project / "autoresearch/06_experiments/data/reltwin_esc50_v1"
    runs = project / "autoresearch/06_experiments/runs"
    base, official = project / "models/audio-flamingo-3-hf", project / "models/SpotSound"
    sft = runs / "reltwin_esc50_v1/sft/adapter"
    parents = [runs / "reltwin_esc50_v1/rbee/adapter"] + [runs / f"e002_matched_pipeline_corrected/seed_{s}/rbee/adapter" for s in (1, 2)]
    setpos = [runs / "reltwin_setpo_warm64_seed0_v1/adapter"] + [runs / f"e002_matched_pipeline_corrected/seed_{s}/setpo/adapter" for s in (1, 2)]
    manifests = {"relation": data / "test.json", "public": project / "datasets/SpotSound-Bench/annotations_processed.json"}
    audios = {"relation": data / "audio", "public": project / "datasets/SpotSound-Bench/audio"}
    for name, n in (("relation", 320), ("public", 400)):
        rows = read(manifests[name])
        if len(rows) != n:
            raise ValueError(f"Unexpected {name} size")
        for row in rows:
            path = audios[name] / Path(row["audio_path"]).name
            if not path.is_file():
                raise FileNotFoundError(path)
    hash_paths = [data / n for n in ("train.json", "test.json", "rehearsal.json")]
    hash_paths += [manifests["public"]]
    for adapter in [official, sft, *parents, *setpos]:
        hash_paths.extend([adapter / "adapter_model.safetensors", adapter / "adapter_config.json"])
    hash_paths += sorted(base.glob("*.json")) + sorted(base.glob("*.safetensors"))
    hash_paths += [repo / "scripts" / name for name in ("run_reltwin_review_controls.py", "train_reltwin_micro.py", "spotsound.py", "evaluate_checkpoint.py", "interval_metrics.py", "analyze_reltwin_probe.py")]
    protocol = repo / "experiments/protocols/RELTWIN_REVIEW_CONTROLS_20260912.md"
    hash_paths.append(protocol)
    frozen = {str(p): sha(p) for p in hash_paths}
    tasks = []

    def add(name, command, artifact, **extra):
        tasks.append({"name": name, "command": [str(v) for v in command], "artifact": str(artifact), **extra})

    def evaluate(label, adapter, smoke=False):
        root = output / label
        for split in ("relation", "public"):
            pred = root / f"{split}_predictions.jsonl"
            summary = root / f"{split}_summary.json"
            cmd = [sys.executable, repo / "scripts/evaluate_checkpoint.py", "--base", base, "--adapter", adapter,
                   "--annotations", manifests[split], "--audio-dir", audios[split], "--predictions", pred, "--summary", summary]
            if smoke:
                cmd += ["--limit", "10"]
            add(f"{label}/{split}", cmd, summary, kind="eval", predictions=str(pred), manifest=str(manifests[split]), count=10 if smoke else None)
            if split == "relation" and not smoke:
                analysis = root / "relation_analysis.json"
                add(f"{label}/relation_analysis", [sys.executable, repo / "scripts/analyze_reltwin_probe.py", "--manifest", manifests[split], "--predictions", pred, "--output", analysis], analysis, kind="analysis")

    def train(label, parent, seed, steps, lr, exchange):
        root = output / label
        adapter, summary = root / "adapter", root / "train_summary.json"
        cmd = [sys.executable, repo / "scripts/train_reltwin_micro.py", "--base", base, "--adapter", parent,
               "--manifest", data / "train.json", "--rehearsal-manifest", data / "rehearsal.json", "--audio-dir", data / "audio",
               "--output-adapter", adapter, "--summary", summary, "--mode", "rbee", "--steps", steps,
               "--learning-rate", lr, "--temperature", 1., "--method-weight", 1., "--exchange-weight", exchange,
               "--rehearsal-weight", .5, "--cache-groups", 512, "--seed", seed]
        add(f"{label}/train", cmd, summary, kind="train", adapter=str(adapter), steps=steps, exchange_weight=exchange)
        return adapter

    evaluate("smoke_rbee_seed0", parents[0], smoke=True)
    for seed in range(3):
        label = f"no_exchange_seed{seed}"
        adapter = train(label, official, seed, 256, 5e-6, 0.)
        evaluate(label, adapter)
        evaluate(f"rbee_seed{seed}", parents[seed])
    for seed in range(3):
        label = f"continue_rbee_seed{seed}"
        adapter = train(label, parents[seed], seed, 64, 2e-6, 1.)
        evaluate(label, adapter)
        evaluate(f"setpo_seed{seed}", setpos[seed])
    evaluate("official", official)
    evaluate("sft_seed0", sft)

    import torch
    import transformers
    import peft
    definition = {"hashes": frozen, "tasks": tasks, "python": sys.version, "torch": torch.__version__,
                  "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0),
                  "platform": platform.platform(), "schema_version": 1}
    freeze_file = output / "freeze.json"
    if freeze_file.exists():
        if read(freeze_file) != definition:
            raise ValueError("Frozen experiment definition changed; do not resume in this output directory")
    else:
        write(freeze_file, definition)
    write(output / "launch.json", {"pid": os.getpid(), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(), "time": time.time()})
    if args.plan_only:
        print(json.dumps({"frozen": str(freeze_file), "tasks": len(tasks)}), flush=True)
        return
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    completed = []

    def update(state, task=None, error=None):
        write(output / "status.json", {"state": state, "pid": os.getpid(), "time": time.time(), "completed": completed, "total_tasks": len(tasks), "current_task": task, "error": error})
        lines = ["# RelTwin review controls — experiment log", "", f"State: {state}. Completed tasks: {len(completed)}/{len(tasks)}.", "", "No new hyperparameter selection; unfavorable results are retained.", "", "| Model | Steps | Exchange | Relation mIoU | PairAcc | Swap error | SpotSound mIoU |", "|---|---:|---:|---:|---:|---:|---:|"]
        for root in sorted(output.iterdir()):
            if not root.is_dir() or root.name.startswith("smoke") or root.name == "logs":
                continue
            t = read(root / "train_summary.json") if (root / "train_summary.json").exists() else {}
            a = read(root / "relation_analysis.json") if (root / "relation_analysis.json").exists() else {}
            p = read(root / "public_summary.json") if (root / "public_summary.json").exists() else {}
            def pct(k): return f"{100*a[k]:.6f}" if k in a else "pending"
            lines.append(f"| {root.name} | {t.get('steps','existing')} | {t.get('exchange_weight','existing')} | {pct('query_mIoU')} | {pct('pair_acc_0.5')} | {pct('swap_error_rate')} | {p.get('mIoU','pending')} |")
        if error:
            lines += ["", f"Failure: {error}"]
        (output / "EXPERIMENT_LOG.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    try:
        for task in tasks:
            # Do not silently change the trained or evaluated method mid-queue.
            for p, expected in frozen.items():
                if str(repo / "scripts") in p and sha(p) != expected:
                    raise RuntimeError(f"Source changed during queue: {p}")
            artifact = Path(task["artifact"])
            update("RUNNING", task["name"])
            if not artifact.exists():
                if task["kind"] == "train" and Path(task["adapter"]).exists():
                    raise RuntimeError(f"Partial adapter save requires manual audit: {task['adapter']}")
                log = output / "logs" / (task["name"].replace("/", "__")+".log")
                log.parent.mkdir(exist_ok=True)
                with log.open("a", encoding="utf-8") as handle:
                    result = subprocess.run(task["command"], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Task {task['name']} exited {result.returncode}; see {log}")
            value = read(artifact)
            if task["kind"] == "train":
                if value["steps"] != task["steps"] or value["exchange_weight"] != task["exchange_weight"]:
                    raise ValueError(f"Training result mismatch: {artifact}")
                if not (Path(task["adapter"]) / "adapter_model.safetensors").is_file():
                    raise FileNotFoundError(task["adapter"])
            elif task["kind"] == "eval":
                rows = validate_predictions(task["predictions"], task["manifest"], task["count"])
                if value["completed_count"] != len(rows):
                    raise ValueError(f"Evaluation summary mismatch: {artifact}")
            elif value["query_count"] != 320 or value["relation_pair_count"] != 160:
                raise ValueError(f"Relation analysis mismatch: {artifact}")
            completed.append(task["name"])
            print(json.dumps({"completed": task["name"], "progress": [len(completed), len(tasks)]}), flush=True)
            update("RUNNING")
        update("COMPLETE")
    except Exception as exc:
        update("FAILED", task.get("name") if "task" in locals() else None, str(exc))
        raise


if __name__ == "__main__":
    main()
