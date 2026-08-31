#!/usr/bin/env python3
"""Test NOVA's load-bearing Keep/Drop signal before training a router."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from nova import counterfactual_features
from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3TemporalProcessor
from train_reltwin_micro import sequence_score


FEATURES = (
    "component_mean",
    "component_min",
    "complement_no",
    "mean_plus_complement",
    "min_plus_complement",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], row["template"], row.get("variant", 0))][
            row["relation"]
        ] = row
    groups = [group for _, group in sorted(grouped.items()) if set(group) == {"AB", "BA"}]
    groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("No complete AB/BA groups found")

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, torch_device="cpu")
    model = model.eval().to("cuda")

    comparisons = []
    for group_index, group in enumerate(groups):
        ab, ba = group["AB"], group["BA"]
        audio_path = args.audio_dir / Path(ab["audio_path"]).name
        wave, _ = librosa.load(audio_path, sr=16000, mono=True)
        wave = np.asarray(wave, dtype=np.float32)
        for relation, row, wrong in (
            ("AB", ab, ba["annotations"]),
            ("BA", ba, ab["annotations"]),
        ):
            correct_features = counterfactual_features(
                model,
                processor,
                wave,
                row["caption"],
                row["annotations"],
                sequence_score,
            )
            wrong_features = counterfactual_features(
                model,
                processor,
                wave,
                row["caption"],
                wrong,
                sequence_score,
            )
            event = {
                "group_index": group_index,
                "pair_id": row["pair_id"],
                "relation": relation,
                "query": row["caption"],
                "correct": correct_features,
                "wrong": wrong_features,
            }
            comparisons.append(event)
            print(json.dumps(event), flush=True)

    accuracies = {
        feature: float(
            np.mean(
                [item["correct"][feature] > item["wrong"][feature] for item in comparisons]
            )
        )
        for feature in FEATURES
    }
    ties = {
        feature: int(
            sum(item["correct"][feature] == item["wrong"][feature] for item in comparisons)
        )
        for feature in FEATURES
    }
    summary = {
        "status": "diagnostic_only",
        "groups": len(groups),
        "comparisons": len(comparisons),
        "accuracies": accuracies,
        "ties": ties,
        "rows": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
