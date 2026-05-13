from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from clickbait_common import (
    RESULTS_DIR,
    SPLIT_INDICES_PATH,
    append_metrics_record,
    get_split,
    load_csv_dataset,
    resolve_project_path,
    sanitize_name,
)
from clickbait_vision import ThumbnailDataset


PROMPTS = ["a non-clickbait YouTube thumbnail", "a clickbait YouTube thumbnail"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP zero-shot probe for thumbnail clickbait detection.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_thumbnails.csv")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--split-indices-path", type=Path, default=SPLIT_INDICES_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc_roc"] = float("nan")
    return metrics


def sanity_check(model, processor, loader, device: torch.device) -> None:
    sample = next(iter(loader))
    image = Image.open(sample["image_path"][0]).convert("RGB")
    inputs = processor(text=PROMPTS, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Sanity check logits shape: {tuple(outputs.logits_per_image.shape)}")


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = resolve_project_path(args.csv)
    dataset = load_csv_dataset(csv_path)

    # Use canonical split so test set matches all other scripts
    split = get_split(dataset, args.split_indices_path, seed=args.seed)
    rows = split[args.split].to_pandas().to_dict(orient="records")

    thumbnail_dataset = ThumbnailDataset(rows, image_size=args.image_size, train=False)
    loader = DataLoader(thumbnail_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    sanity_check(model, processor, loader, device)

    all_logits = []
    labels = []
    for batch in tqdm(loader, desc=f"CLIP zero-shot ({args.split})"):
        images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
        inputs = processor(text=PROMPTS, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits_per_image.detach().cpu().numpy())
        labels.extend(batch["labels"].numpy().tolist())

    logits = np.concatenate(all_logits, axis=0)
    metrics = compute_metrics(logits, np.asarray(labels, dtype=int))
    results_path = resolve_project_path(args.results_path)
    append_metrics_record(results_path, {
        "model": sanitize_name(args.model_name),
        "stage": f"clip_zero_shot_{args.split}",
        "metrics": metrics,
    })
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
