#!/usr/bin/env python3
"""Remove the layout-correlated silence pattern; run two frozen models, no training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from run_reltwin_review_controls import read, write, sha, validate_predictions


def uniform_timing(wave, sr, row):
    if sr != 16000:
        raise ValueError("Expected the frozen 16 kHz PCM audio")
    a0, a1 = row["window_ab"][0]
    b0, b1 = row["window_ba"][0]
    n = 5 * sr
    a = wave[round(a0 * sr):round(a0 * sr) + n]
    b = wave[round(a1 * sr) - n:round(a1 * sr)]
    if not np.array_equal(b, wave[round(b0 * sr):round(b0 * sr) + n]) or not np.array_equal(a, wave[round(b1 * sr) - n:round(b1 * sr)]):
        raise ValueError("Constituent durations/duplicate samples do not match")
    zero = lambda seconds: np.zeros(round(seconds * sr), dtype=wave.dtype)
    ab, ba = np.concatenate([a, zero(.25), b]), np.concatenate([b, zero(.25), a])
    first, second = (ab, ba) if row["layout"] == "AB_first" else (ba, ab)
    new = np.concatenate([zero(1.5), first, zero(3), second, zero(2)])
    if row["layout"] == "AB_first" and not np.array_equal(wave, new):
        raise ValueError("Unchanged-layout identity check failed")
    windows = {"AB": [[1.5, 11.75]], "BA": [[14.75, 25.]]}
    if row["layout"] == "BA_first":
        windows = {"BA": windows["AB"], "AB": windows["BA"]}
    return new, windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    project, output = args.project_root.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bridge = project / "outputs/reltwin_paper_bridge_20260913"
    controls = project / "outputs/reltwin_review_controls_20260912"
    data = project / "autoresearch/06_experiments/data/reltwin_esc50_v1"
    source = read(data / "test.json")
    selected = [(i, row) for i, row in enumerate(source) if row["template"] == "followed_by"]
    if len(selected) != 160:
        raise ValueError("Unexpected predefined template count")
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest, files, rebuilt = [], [data / "test.json"], {}
    for index, row in selected:
        name = Path(row["audio_path"]).name
        if name not in rebuilt:
            original = data / "audio" / name
            wave, sr = sf.read(original, dtype="float32")
            new, windows = uniform_timing(wave, sr, row)
            target = audio_dir / name
            if target.exists():
                saved, rate = sf.read(target, dtype="float32")
                if rate != sr or not np.array_equal(saved, new):
                    raise ValueError("Existing stress audio differs")
            else:
                sf.write(target, new, sr, subtype="PCM_16")
                saved, _ = sf.read(target, dtype="float32")
                assert np.array_equal(saved, new)
            files += [original, target]
            rebuilt[name] = windows
        windows = rebuilt[name]
        manifest.append({**row, "audio_path": str(audio_dir / name), "source_index": index, "annotations": windows[row["relation"]], "window_ab": windows["AB"], "window_ba": windows["BA"], "stress_factor": "layout_neutral_timing"})
    assert len(rebuilt) == 80
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Stress manifest changed")
    write(manifest_path, manifest)
    adapters = {"sft_current_seed0": bridge / "sft_current_seed0/adapter", "no_exchange_seed0": controls / "no_exchange_seed0/adapter"}
    base = project / "models/audio-flamingo-3-hf"
    files += [manifest_path, repo / "experiments/protocols/RELTWIN_TIMING_STRESS_20260913.md", Path(__file__).resolve()]
    files += [repo / "scripts" / n for n in ("evaluate_checkpoint.py", "spotsound.py", "interval_metrics.py", "run_reltwin_review_controls.py")]
    files += [p for a in adapters.values() for p in (a / "adapter_model.safetensors", a / "adapter_config.json")]
    tasks = []
    for name, adapter in adapters.items():
        root = output / name
        pred, summary = root / "predictions.jsonl", root / "summary.json"
        tasks.append({"name": name, "predictions": str(pred), "summary": str(summary), "command": [str(x) for x in [sys.executable, repo / "scripts/evaluate_checkpoint.py", "--base", base, "--adapter", adapter, "--annotations", manifest_path, "--audio-dir", audio_dir, "--predictions", pred, "--summary", summary]]})
    definition = {"budget_seconds": 3600, "hashes": {str(p): sha(p) for p in sorted(set(files))}, "tasks": tasks, "base_runtime_freeze": sha(bridge / "freeze.json"), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    freeze = output / "freeze.json"
    if freeze.exists():
        old = read(freeze)
        definition["code_commit"] = old["code_commit"]
        if old != definition:
            raise ValueError("Frozen stress definition changed")
    else:
        write(freeze, definition)
    if args.prepare_only:
        print(json.dumps({"queries": len(manifest), "audio_groups": len(rebuilt), "models": list(adapters), "frozen": str(freeze)}))
        return
    if read(bridge / "status.json")["state"] != "COMPLETE":
        raise RuntimeError("Wait for bridge completion; never compete on the GPU")
    status_path = output / "status.json"
    previous = read(status_path) if status_path.exists() else {}
    if previous.get("state") == "RUNNING":
        raise RuntimeError("Interrupted stress process requires budget audit before resume")
    used, completed = previous.get("used_seconds", 0.), []
    def status(state, name=None, error=None):
        write(status_path, {"state": state, "pid": os.getpid(), "time": time.time(), "current_task": name, "completed": completed, "total_tasks": 2, "used_seconds": used, "error": error})
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    task = None
    try:
        for task in tasks:
            status("RUNNING", task["name"])
            if not Path(task["summary"]).exists():
                remaining = 3600 - used
                if remaining <= 0:
                    raise TimeoutError("Stress budget exhausted")
                began = time.monotonic()
                try:
                    with (output / f"{task['name']}.log").open("a", encoding="utf-8") as handle:
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
                            raise RuntimeError(f"Evaluation failed: {task['name']}")
                finally:
                    used += time.monotonic() - began
            rows = validate_predictions(task["predictions"], manifest_path)
            assert read(task["summary"])["completed_count"] == len(rows) == 160
            completed.append(task["name"])
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED", task["name"] if task else None, str(exc))
        raise


if __name__ == "__main__":
    main()
