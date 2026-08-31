#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def interval_union_length(intervals):
    merged = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s)):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(e - s for s, e in merged)


def set_iou(left, right):
    points = sorted({float(x) for interval in left + right for x in interval})
    intersection = union = 0.0
    for a, b in zip(points, points[1:]):
        middle = (a + b) / 2
        in_left = any(s <= middle <= e for s, e in left)
        in_right = any(s <= middle <= e for s, e in right)
        if in_left or in_right:
            union += b - a
        if in_left and in_right:
            intersection += b - a
    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    predictions = [json.loads(x) for x in Path(args.predictions).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(manifest) == len(predictions)

    rows = []
    groups = defaultdict(dict)
    for item, pred in zip(manifest, predictions):
        prediction = pred.get("prediction", [])
        correct = item["window_ab"] if item["relation"] == "AB" else item["window_ba"]
        wrong = item["window_ba"] if item["relation"] == "AB" else item["window_ab"]
        iou_correct = set_iou(prediction, correct)
        iou_wrong = set_iou(prediction, wrong)
        chosen = "correct" if iou_correct > iou_wrong else "wrong" if iou_wrong > iou_correct else "tie"
        row = {
            "index": pred["index"],
            "pair_id": item["pair_id"],
            "template": item["template"],
            "relation": item["relation"],
            "query": item["caption"],
            "prediction": prediction,
            "raw_answer": pred.get("raw_answer"),
            "iou_correct": iou_correct,
            "iou_wrong": iou_wrong,
            "chosen": chosen,
        }
        rows.append(row)
        groups[(item["pair_id"], item["template"], item.get("variant", 0))][item["relation"]] = row

    pair_rows = []
    for (pair_id, template, variant), values in sorted(groups.items()):
        ab, ba = values["AB"], values["BA"]
        pred_similarity = set_iou(ab["prediction"], ba["prediction"])
        pair_rows.append({
            "pair_id": pair_id,
            "template": template,
            "variant": variant,
            "pair_acc_0.3": ab["iou_correct"] >= 0.3 and ba["iou_correct"] >= 0.3,
            "pair_acc_0.5": ab["iou_correct"] >= 0.5 and ba["iou_correct"] >= 0.5,
            "swap_error": ab["chosen"] != "correct" or ba["chosen"] != "correct",
            "prediction_set_iou_under_query_swap": pred_similarity,
            "ab_choice": ab["chosen"],
            "ba_choice": ba["chosen"],
        })

    result = {
        "query_count": len(rows),
        "relation_pair_count": len(pair_rows),
        "query_R1@0.3": sum(x["iou_correct"] >= 0.3 for x in rows) / len(rows),
        "query_R1@0.5": sum(x["iou_correct"] >= 0.5 for x in rows) / len(rows),
        "query_mIoU": sum(x["iou_correct"] for x in rows) / len(rows),
        "wrong_window_preference_rate": sum(x["chosen"] == "wrong" for x in rows) / len(rows),
        "pair_acc_0.3": sum(x["pair_acc_0.3"] for x in pair_rows) / len(pair_rows),
        "pair_acc_0.5": sum(x["pair_acc_0.5"] for x in pair_rows) / len(pair_rows),
        "swap_error_rate": sum(x["swap_error"] for x in pair_rows) / len(pair_rows),
        "mean_prediction_set_iou_under_query_swap": sum(x["prediction_set_iou_under_query_swap"] for x in pair_rows) / len(pair_rows),
        "pair_rows": pair_rows,
        "query_rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("rows")}, indent=2))


if __name__ == "__main__":
    main()
