from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from clickbait_common import (
    RESULTS_DIR,
    SPLIT_INDICES_PATH,
    append_metrics_record,
    chat_completion,
    compute_binary_metrics,
    get_split,
    load_csv_dataset,
    parse_json_object,
    resolve_project_path,
    sanitize_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenRouter zero-shot clickbait classification for titles.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_titles.csv")
    parser.add_argument("--title-column", type=str, default="title")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--referer", type=str, default="")
    parser.add_argument("--app-name", type=str, default="clickbait-zero-shot")
    parser.add_argument("--output-predictions", type=Path, default=Path("results") / "openrouter_predictions.csv")
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--split-indices-path", type=Path, default=SPLIT_INDICES_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def classify_title(*, base_url, api_key, model, title, referer, app_name) -> tuple[int, float, str]:
    system_prompt = (
        "You are a strict binary classifier for YouTube titles. "
        "Return JSON with keys 'label' (0 for non-clickbait, 1 for clickbait) and 'confidence' between 0 and 1."
    )
    raw = chat_completion(
        base_url=base_url, api_key=api_key, model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this title:\n{title}"},
        ],
        max_tokens=64, temperature=0.0,
        extra_headers={k: v for k, v in {
            "HTTP-Referer": referer or None, "X-Title": app_name or None
        }.items() if v},
    )
    try:
        payload = parse_json_object(raw)
        return int(payload.get("label", 0)), float(payload.get("confidence", 0.5)), raw
    except Exception:
        lowered = raw.lower()
        return (1 if "clickbait" in lowered and "non" not in lowered else 0), 0.5, raw


def main() -> None:
    args = build_parser().parse_args()
    csv_path = resolve_project_path(args.csv)
    output_predictions = resolve_project_path(args.output_predictions)
    results_path = resolve_project_path(args.results_path)

    dataset = load_csv_dataset(csv_path)

    # Use canonical split so test set matches all other scripts
    split = get_split(dataset, args.split_indices_path, seed=args.seed)
    rows = split[args.split].to_pandas().to_dict(orient="records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    predictions, labels, logits = [], [], []
    output_predictions.parent.mkdir(parents=True, exist_ok=True)

    with output_predictions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "videoID", "is_clickbait", "title", "predicted_label", "confidence", "raw_response"
        ])
        writer.writeheader()

        for row in tqdm(rows, desc=f"OpenRouter zero-shot ({args.split})"):
            title = str(row.get(args.title_column) or "").strip()
            predicted_label, confidence, raw_response = classify_title(
                base_url=args.base_url, api_key=args.api_key, model=args.model,
                title=title, referer=args.referer, app_name=args.app_name,
            )
            label = int(row.get("is_clickbait", 0))
            labels.append(label)
            predictions.append(predicted_label)
            logits.append(
                [1.0 - confidence, confidence] if predicted_label == 1
                else [confidence, 1.0 - confidence]
            )
            writer.writerow({
                "videoID": row.get("videoID", ""), "is_clickbait": label,
                "title": title, "predicted_label": predicted_label,
                "confidence": confidence, "raw_response": raw_response,
            })

    metrics = compute_binary_metrics(np.asarray(logits, dtype=float), np.asarray(labels, dtype=int))
    append_metrics_record(results_path, {
        "model": sanitize_name(args.model),
        "stage": f"zero_shot_{args.split}",
        "metrics": metrics,
    })
    print(f"Metrics: {metrics}")
    print(f"Saved predictions to {output_predictions}")


if __name__ == "__main__":
    main()
