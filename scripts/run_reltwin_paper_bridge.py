#!/usr/bin/env python3
"""One frozen SFT runtime bridge, bounded wall-time, on the existing single GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from run_reltwin_review_controls import read, write, sha, validate_predictions


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
    controls = project / "outputs/reltwin_review_controls_20260912"
    old = read(controls / "freeze.json")
    base, initial = project / "models/audio-flamingo-3-hf", project / "models/SpotSound"
    manifests = {"relation": data / "test.json", "public": project / "datasets/SpotSound-Bench/annotations_processed.json"}
    audios = {"relation": data / "audio", "public": project / "datasets/SpotSound-Bench/audio"}
    root = output / "sft_current_seed0"
    adapter = root / "adapter"
    files = [repo / "experiments/protocols/RELTWIN_PAPER_POLISH_20260913.md", Path(__file__).resolve()]
    for name in ("train_reltwin_micro.py", "evaluate_checkpoint.py", "spotsound.py", "interval_metrics.py", "analyze_reltwin_probe.py", "run_reltwin_review_controls.py"):
        path = repo / "scripts" / name
        old_hashes = [v for k, v in old["hashes"].items() if k.endswith("/scripts/" + name)]
        if old_hashes != [sha(path)]:
            raise ValueError(f"Current source differs from existing control: {name}")
        files.append(path)
    for name, directory in (("train.json", data / "audio"), ("rehearsal.json", data / "audio"), ("test.json", data / "audio")):
        path = data / name
        if sha(path) != old["hashes"][str(path)]:
            raise ValueError(f"Manifest changed: {path}")
        files.append(path)
        files.extend(directory / Path(row["audio_path"]).name for row in read(path))
    for split, size in (("relation", 320), ("public", 400)):
        manifest = manifests[split]
        if len(read(manifest)) != size or sha(manifest) != old["hashes"][str(manifest)]:
            raise ValueError(f"Evaluation input changed: {split}")
        files.append(manifest)
        files.extend(audios[split] / Path(row["audio_path"]).name for row in read(manifest))
        files.append(controls / "no_exchange_seed0" / f"{split}_predictions.jsonl")
    files += sorted(base.glob("*.json")) + sorted(base.glob("*.safetensors"))
    files += [initial / "adapter_model.safetensors", initial / "adapter_config.json", controls / "no_exchange_seed0/train_summary.json"]
    reference = read(controls / "no_exchange_seed0/train_summary.json")
    expected = {"steps": 256, "learning_rate": 5e-6, "seed": 0, "rehearsal_weight": .5, "exchange_weight": 0., "relation_pairs": 512, "rehearsal_examples": 512}
    if any(reference[k] != v for k, v in expected.items()):
        raise ValueError("Existing control is not the expected matched configuration")
    tasks = [{"name": "train", "kind": "train", "artifact": str(root / "train_summary.json"), "command": [str(x) for x in [sys.executable, repo / "scripts/train_reltwin_micro.py", "--base", base, "--adapter", initial, "--manifest", data / "train.json", "--rehearsal-manifest", data / "rehearsal.json", "--audio-dir", data / "audio", "--output-adapter", adapter, "--summary", root / "train_summary.json", "--mode", "sft", "--steps", 256, "--learning-rate", 5e-6, "--temperature", 1., "--method-weight", 1., "--exchange-weight", 0., "--rehearsal-weight", .5, "--cache-groups", 512, "--seed", 0]]}]
    for smoke in (True, False):
        for split in ("relation", "public"):
            prefix = f"smoke_{split}" if smoke else split
            pred, summary = root / f"{prefix}_predictions.jsonl", root / f"{prefix}_summary.json"
            cmd = [sys.executable, repo / "scripts/evaluate_checkpoint.py", "--base", base, "--adapter", adapter, "--annotations", manifests[split], "--audio-dir", audios[split], "--predictions", pred, "--summary", summary]
            if smoke:
                cmd += ["--limit", 10]
            tasks.append({"name": prefix, "kind": "eval", "command": [str(x) for x in cmd], "artifact": str(summary), "predictions": str(pred), "manifest": str(manifests[split]), "count": 10 if smoke else None})
    analysis = root / "relation_analysis.json"
    tasks.append({"name": "relation_analysis", "kind": "analysis", "artifact": str(analysis), "command": [str(x) for x in [sys.executable, repo / "scripts/analyze_reltwin_probe.py", "--manifest", manifests["relation"], "--predictions", root / "relation_predictions.jsonl", "--output", analysis]]})
    import torch
    import transformers
    import peft
    runtime = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0), "platform": platform.platform()}
    for key in runtime:
        if runtime[key] != old[key]:
            raise ValueError(f"Training environment changed: {key}")
    definition = {"schema": 1, "runtime": runtime, "hashes": {str(p): sha(p) for p in sorted(set(files))}, "tasks": tasks, "budget_seconds": 7200, "reference_freeze_hash": sha(controls / "freeze.json")}
    freeze = output / "freeze.json"
    if freeze.exists() and read(freeze) != definition:
        raise ValueError("Frozen definition changed; refusing resume")
    if not freeze.exists():
        write(freeze, definition)
    write(output / "launch.json", {"pid": os.getpid(), "time": time.time(), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()})
    if args.plan_only:
        print(json.dumps({"tasks": len(tasks), "frozen_files": len(definition["hashes"])}), flush=True)
        return
    ledger_path = output / "budget.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"used_seconds": 0., "active_started": None}
    if ledger["active_started"] is not None:
        # A previous machine/queue crash needs inspection before spending more budget.
        raise RuntimeError("Unclosed budget ledger: audit interrupted task before resuming")
    completed = []
    def status(state, task=None, error=None):
        write(output / "status.json", {"state": state, "pid": os.getpid(), "time": time.time(), "completed": completed, "total_tasks": len(tasks), "current_task": task, "error": error, "used_seconds": ledger["used_seconds"]})
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    task = None
    try:
        for task in tasks:
            for path, digest in definition["hashes"].items():
                if str(repo) in path and sha(path) != digest:
                    raise RuntimeError(f"Frozen source changed: {path}")
            status("RUNNING", task["name"])
            artifact = Path(task["artifact"])
            if not artifact.exists():
                if task["kind"] == "train" and adapter.exists():
                    raise RuntimeError("Partial adapter: manual audit required, no overwrite")
                remaining = definition["budget_seconds"] - ledger["used_seconds"]
                if remaining <= 0:
                    raise TimeoutError("Cumulative budget exhausted")
                log = output / "logs" / f"{task['name']}.log"
                log.parent.mkdir(exist_ok=True)
                ledger["active_started"] = time.time()
                write(ledger_path, ledger)
                began = time.monotonic()
                try:
                    with log.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(task["command"], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            result = process.wait(timeout=remaining)
                        except BaseException:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise
                        if result:
                            raise RuntimeError(f"Task {task['name']} exited {result}; see {log}")
                finally:
                    ledger["used_seconds"] += time.monotonic() - began
                    ledger["active_started"] = None
                    write(ledger_path, ledger)
            value = read(artifact)
            if task["kind"] == "train":
                if value["mode"] != "sft" or any(value[k] != v for k, v in expected.items()):
                    raise ValueError("Training configuration mismatch")
                if not (adapter / "adapter_model.safetensors").is_file():
                    raise FileNotFoundError(adapter)
                write(root / "adapter_hashes.json", {p.name: sha(p) for p in sorted(adapter.iterdir()) if p.is_file()})
            elif task["kind"] == "eval":
                rows = validate_predictions(task["predictions"], task["manifest"], task["count"])
                if value["completed_count"] != len(rows):
                    raise ValueError("Summary count mismatch")
            elif value["query_count"] != 320 or value["relation_pair_count"] != 160:
                raise ValueError("Relation count mismatch")
            completed.append(task["name"])
            print(json.dumps({"completed": task["name"], "progress": [len(completed), len(tasks)]}), flush=True)
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED", task["name"] if task else None, str(exc))
        raise


if __name__ == "__main__":
    main()
