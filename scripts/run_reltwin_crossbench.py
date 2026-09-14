#!/usr/bin/env python3
"""Durable inference-only queue; never gate later benchmarks on measured scores."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reltwin_crossbench_support import read, write, sha, load_rows, validate_rows, summarize, paired_summary

REPO = Path(__file__).resolve().parents[1]
SPECS = [
    ("unav", "UnAV-100", "annotations_processed.json", "audio", 100, "spotsound", 128, False),
    ("tut", "TUT2017", "annotations_processed.json", "audio", 104, "spotsound", 128, False),
    ("audiogrounding", "AudioGrounding-v2", "annotations_processed.json", "official_audio", 997, "spotsound", 128, False),
    ("desed", "DESED-public-eval", "annotations_processed.json", "extracted/dataset/audio/eval/public", 1112, "spotsound", 128, False),
    ("clotho", "Clotho-Moment", "annotations_processed.json", "audio", 6649, "spotsound", 128, False),
    ("aegbench", "AEGBench", "annotations_processed.json", ".", 9924, "aegbench", 256, False),
    ("lat", "LAT-Bench", "annotations_en_tag.json", "audios", 426, "spotsound", 128, True),
]


def prepare(project, output):
    import soundfile as sf
    import torch
    import transformers
    import peft
    controls = project / "outputs/reltwin_review_controls_20260912"
    adapters = {"reltwin_seed0": controls / "no_exchange_seed0/adapter",
                "sft_seed0": project / "outputs/reltwin_paper_bridge_20260913/sft_current_seed0/adapter"}
    base = project / "models/audio-flamingo-3-hf"
    # Verify the unchanged inference implementation against the paper's execution freeze.
    old = read(controls / "freeze.json")
    frozen = {}
    for name in ("evaluate_checkpoint.py", "spotsound.py", "interval_metrics.py"):
        path = REPO / "scripts" / name
        expected = [v for k, v in old["hashes"].items() if k.endswith("/scripts/" + name)]
        if expected != [sha(path)]:
            raise ValueError(f"Inference source differs from paper execution: {name}")
    for model, adapter in adapters.items():
        summary = read(adapter.parent / "train_summary.json")
        if summary["seed"] != 0 or summary["steps"] != 256 or summary["exchange_weight"] != 0:
            raise ValueError(f"Wrong adapter identity: {model}")
        for path in adapter.glob("*"):
            if path.is_file():
                frozen[str(path)] = sha(path)
    for path in list(base.glob("*.json")) + list(base.glob("*.safetensors")):
        frozen[str(path)] = sha(path)
    for path in (REPO / "scripts").glob("*.py"):
        frozen[str(path)] = sha(path)
    specs = []
    audio_inventory = {}
    for key, directory, annotation, audio_dir, count, prompt, tokens, long in SPECS:
        root = project / "datasets" / directory
        source = root / annotation
        rows = read(source)
        if len(rows) != count:
            raise ValueError(f"Unexpected {key} cardinality: {len(rows)} != {count}")
        manifest = []
        for i, row in enumerate(rows):
            rel = Path(row["audio_path"])
            candidates = [root / audio_dir / rel, root / audio_dir / rel.name]
            candidates += [p.with_suffix(s) for p in list(candidates) for s in (".wav", ".flac", ".mp3")]
            path = next((p.resolve() for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"{key}/{i}: {row['audio_path']}")
            if str(path) not in audio_inventory:
                info = sf.info(path)
                audio_inventory[str(path)] = {"frames": info.frames, "sample_rate": info.samplerate,
                                             "duration": info.frames / info.samplerate, "bytes": path.stat().st_size,
                                             "sha256": sha(path)}
            duration = audio_inventory[str(path)]["duration"]
            if duration <= 0 or (not long and duration > 600.0 + 1e-6):
                raise ValueError(f"Unsupported duration {key}/{i}: {duration}")
            manifest.append(dict(row, audio_path=str(path), source_index=i,
                                 audio_group=str(path), decoded_duration=duration))
        destination = output / "manifests" / f"{key}.json"
        write(destination, manifest)
        frozen[str(source)] = sha(source)
        frozen[str(destination)] = sha(destination)
        specs.append(dict(key=key, source=str(source), manifest=str(destination), count=count,
                          prompt=prompt, max_new_tokens=tokens, long=long))
        print(f"Prepared {key}: {count} queries", flush=True)
    inventory = output / "audio_inventory.json"
    write(inventory, audio_inventory)
    frozen[str(inventory)] = sha(inventory)
    tasks = [dict(benchmark=s["key"], model="reltwin_seed0") for s in specs]
    # These paired controls answer whether transfer is specific to the candidate objective.
    # Their order and inclusion are fixed now, before any new full scores are inspected.
    tasks += [dict(benchmark=key, model="sft_seed0") for key in ("unav", "audiogrounding", "clotho")]
    plan = {"schema": 1, "project": str(project), "base": str(base),
            "adapters": {k: str(v) for k, v in adapters.items()}, "specs": specs, "tasks": tasks,
            "hashes": frozen, "training_seed": 0, "inference_seed": 0,
            "runtime": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "no_score_gate": True, "no_training": True, "no_nova_or_boundary_refinement": True,
            "lat_policy": "Same existing 600-second / 300-second-stride label-free existence-ranked window wrapper; report separately",
            "protocol_notes": {"aegbench": "public v3 9924 positive queries; not the historical 9790-query leaderboard",
                               "unav_audiogrounding": "Complete upstream releases; retain documented SpotSound Table 2 count inconsistency",
                               "clotho": "Previously inspected first-1000 seed-0 pilot; no score gate in this run"}}
    write(output / "freeze.json", plan)
    return plan


def verify_code(plan):
    for path, expected in plan["hashes"].items():
        if str(REPO) in path or "/manifests/" in path or "/adapter/" in path:
            if sha(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")


def import_cache(project, output, plan, spec, model):
    target = output / model / spec["key"] / "predictions.jsonl"
    if target.exists():
        return
    candidates = []
    if spec["key"] == "unav" and model == "reltwin_seed0":
        candidates.append(output / "smoke_unav")
    if spec["key"] == "clotho":
        label = "reltwin" if model == "reltwin_seed0" else "sft"
        candidates.append(project / "spotsound_reltwin_polish_20260913/results/reltwin_real_bench_20260914" / label)
    for directory in candidates:
        source, summary = directory / "predictions.jsonl", directory / "summary.json"
        if not source.exists() or not summary.exists():
            continue
        info = read(summary)
        if Path(info["adapter"]).resolve() != Path(plan["adapters"][model]).resolve():
            raise ValueError("Cached adapter identity mismatch")
        try:
            rows = validate_rows(load_rows(source), read(spec["manifest"]))
        except ValueError as exc:
            write(target.parent / "cache_rejected.json", {"source": str(source), "reason": str(exc),
                                                          "action": "preserve source and evaluate fresh"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        write(target.parent / "cache_import.json", {"source": str(source), "sha256": sha(source),
                                                     "rows": len(rows), "identity_validated": True})
        return


def update(output, plan, state, completed, failed, task=None):
    write(output / "status.json", {"state": state, "pid": os.getpid(), "time": time.time(), "current": task,
                                    "completed": completed, "failed": failed, "total": len(plan["tasks"])})
    results = []
    lines = ["# Frozen RelTwin cross-benchmark evaluation", "", f"State: {state}", "",
             "Fixed seed 0; no training, score-based stopping, NOVA routing or boundary edits.",
             "SpotSound's published three-seed mean is not relabeled as a single-seed result.", "",
             "| Benchmark | Model | Rows | mIoU | R1@.3 | R1@.5 | R1@.7 |", "|---|---|---:|---:|---:|---:|---:|"]
    for task_info in plan["tasks"]:
        key, model = task_info["benchmark"], task_info["model"]
        path = output / model / key / "metrics.json"
        if path.exists():
            score = read(path)
            results.append(dict(benchmark=key, model=model, **score))
            lines.append(f"| {key} | {model} | {score['count']} | {score['mIoU']:.6f} | {score['R1@0.3']:.6f} | {score['R1@0.5']:.6f} | {score['R1@0.7']:.6f} |")
        else:
            lines.append(f"| {key} | {model} | pending | — | — | — | — |")
    lines += ["", "AEGBench additionally reports its mean-best-IoU metric; set-IoU is not its leaderboard mIoU.",
              "LAT uses the declared long-context wrapper and is not a direct full-recording inference result.",
              "AudioGrounding/UnAV use complete upstream releases with the documented Table 2 cardinality qualification.",
              "All scores and regressions are retained. Per-run prediction rows, hashes and paired CIs accompany this log."]
    if failed:
        lines += ["", "Failures: " + json.dumps(failed, ensure_ascii=False)]
    (output / "EXPERIMENT_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(output / "summary.json", {"state": state, "results": results, "failed": failed,
                                     "code_commit": plan["code_commit"], "freeze_sha256": sha(output / "freeze.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = output / "freeze.json"
    if args.mode == "prepare":
        if freeze.exists():
            raise FileExistsError("Do not overwrite frozen configuration; use run to resume")
        plan = prepare(project, output)
        update(output, plan, "PREPARED", [], [])
        return
    plan = read(freeze)
    verify_code(plan)
    completed, failed = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH=str(REPO / "scripts"))
    specs = {s["key"]: s for s in plan["specs"]}
    for task in plan["tasks"]:
        key, model = task["benchmark"], task["model"]
        name = f"{model}/{key}"
        spec = specs[key]
        root = output / model / key
        root.mkdir(parents=True, exist_ok=True)
        update(output, plan, "RUNNING", completed, failed, name)
        try:
            verify_code(plan)
            manifest = read(spec["manifest"])
            import_cache(project, output, plan, spec, model)
            predictions = root / "predictions.jsonl"
            rows = validate_rows(load_rows(predictions), manifest)
            if len(rows) != len(manifest):
                script = "evaluate_long_audio_checkpoint.py" if spec["long"] else "evaluate_checkpoint.py"
                command = [sys.executable, str(REPO / "scripts" / script), "--base", plan["base"],
                           "--adapter", plan["adapters"][model], "--annotations", spec["manifest"],
                           "--audio-dir", str(project / "datasets"), "--predictions", str(predictions),
                           "--summary", str(root / "inference_summary.json"), "--max-new-tokens", str(spec["max_new_tokens"])]
                if not spec["long"]:
                    command += ["--prompt-mode", spec["prompt"]]
                write(root / "command.json", command)
                with (root / "inference.log").open("a", encoding="utf-8") as log:
                    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = validate_rows(load_rows(predictions), manifest, complete=True)
            score = summarize(rows, manifest)
            score.update(adapter_sha256=sha(Path(plan["adapters"][model]) / "adapter_model.safetensors"),
                         input_manifest_sha256=sha(spec["manifest"]), predictions_sha256=sha(predictions),
                         code_commit=plan["code_commit"], model=model, benchmark=key, training_seed=0)
            write(root / "metrics.json", score)
            if model == "sft_seed0":
                ours_path = output / "reltwin_seed0" / key / "predictions.jsonl"
                if (output / "reltwin_seed0" / key / "metrics.json").exists():
                    write(output / "comparisons" / f"{key}.json", paired_summary(load_rows(ours_path), rows, manifest))
            completed.append(name)
        except Exception as exc:
            failed.append({"task": name, "error": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)
        update(output, plan, "RUNNING", completed, failed)
    update(output, plan, "COMPLETE" if not failed else "COMPLETE_WITH_FAILURES", completed, failed)


if __name__ == "__main__":
    main()
