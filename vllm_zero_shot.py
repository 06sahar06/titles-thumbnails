from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from clickbait_common import RESULTS_DIR, append_metrics_record, chat_completion, compute_binary_metrics, image_to_data_url, load_csv_dataset, parse_json_object, resolve_project_path, sanitize_name, stratified_split
from clickbait_vision import ThumbnailDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM multimodal zero-shot clickbait classification.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_thumbnails.csv")
    parser.add_argument("--title-column", type=str, default="title")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--output-predictions", type=Path, default=Path("results") / "vllm_predictions.csv")
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def classify_sample(*, base_url: str, api_key: str, model: str, title: str, image_path: Path) -> tuple[int, float, str]:
    system_prompt = (
        "You are a strict binary classifier for YouTube thumbnails and titles. "
        "Return JSON with keys 'label' (0 for non-clickbait, 1 for clickbait) and 'confidence' between 0 and 1."
    )
    user_prompt = f"Title: {title}\nClassify the thumbnail and title combination."
    raw = chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
        max_tokens=64,
        temperature=0.0,
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
    dataset = load_csv_dataset(csv_path)
    split = stratified_split(dataset)
    records = split[args.split].to_pandas().to_dict(orient="records")
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    predictions = []
    labels = []
    logits = []

    results_path = resolve_project_path(args.results_path)
    output_predictions = resolve_project_path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)

    with output_predictions.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["videoID", "is_clickbait", "title", "predicted_label", "confidence", "raw_response"])
        writer.writeheader()

        for row in tqdm(records, desc=f"vLLM zero-shot ({args.split})"):
            thumbnail_dataset = ThumbnailDataset([row], image_size=args.image_size, train=False)
            item = thumbnail_dataset[0]
            title = str(row.get(args.title_column) or "")
            predicted_label, confidence, raw_response = classify_sample(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                title=title,
                image_path=Path(item["image_path"]),
            )
            label = int(row.get("is_clickbait", 0))
            labels.append(label)
            predictions.append(predicted_label)
            logits.append([1.0 - confidence, confidence] if predicted_label == 1 else [confidence, 1.0 - confidence])
            writer.writerow(
                {
                    "videoID": row.get("videoID", ""),
                    "is_clickbait": label,
                    "title": title,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "raw_response": raw_response,
                }
            )

    metrics = compute_binary_metrics(np.asarray(logits, dtype=float), np.asarray(labels, dtype=int))
    append_metrics_record(
        results_path,
        {
            "model": sanitize_name(args.model),
            "stage": f"vllm_zero_shot_{args.split}",
            "metrics": metrics,
        },
    )
    print(f"Metrics: {metrics}")
    print(f"Saved predictions to {output_predictions}")


if __name__ == "__main__":
    main()