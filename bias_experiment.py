from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clickbait_common import RESULTS_DIR, append_metrics_record, load_csv_dataset, resolve_project_path, sanitize_name, stratified_split
from train_multimodal_fusion import evaluate_head, extract_logits, load_image_artifact, load_text_artifacts, make_loader, sanity_check_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure prediction shifts after title neutralization.")
    parser.add_argument("--original-csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_thumbnails.csv")
    parser.add_argument("--neutralized-csv", type=Path, required=True)
    parser.add_argument("--text-base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--text-checkpoint", type=Path, default=Path("checkpoints") / "distilbert_lora")
    parser.add_argument("--image-checkpoint", type=Path, default=Path("checkpoints") / "resnet50_thumbnail" / "best_model.pt")
    parser.add_argument("--fusion-checkpoint", type=Path, default=Path("checkpoints") / "multimodal_fusion" / "fusion_head.pt")
    parser.add_argument("--title-column-neutralized", type=str, default="neutral_title")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    return parser


def load_fusion_head(checkpoint_path: Path, device: torch.device, input_dim: int = 4):
    from torch import nn

    head = nn.Linear(input_dim, 2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()
    return head


def summarize_shift(original_logits: np.ndarray, neutral_logits: np.ndarray) -> dict[str, float]:
    original_prob = torch.softmax(torch.tensor(original_logits), dim=1).numpy()[:, 1]
    neutral_prob = torch.softmax(torch.tensor(neutral_logits), dim=1).numpy()[:, 1]
    original_pred = (original_prob >= 0.5).astype(int)
    neutral_pred = (neutral_prob >= 0.5).astype(int)
    return {
        "mean_clickbait_probability_original": float(original_prob.mean()),
        "mean_clickbait_probability_neutralized": float(neutral_prob.mean()),
        "mean_probability_delta": float((neutral_prob - original_prob).mean()),
        "flip_rate": float((original_pred != neutral_pred).mean()),
        "clickbait_to_non_clickbait_rate": float(((original_pred == 1) & (neutral_pred == 0)).mean()),
        "non_clickbait_to_clickbait_rate": float(((original_pred == 0) & (neutral_pred == 1)).mean()),
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_dataset = load_csv_dataset(resolve_project_path(args.original_csv))
    neutralized_dataset = load_csv_dataset(resolve_project_path(args.neutralized_csv))

    split = stratified_split(original_dataset)
    neutralized_rows = {str(row["videoID"]): row for row in neutralized_dataset.to_pandas().to_dict(orient="records")}

    def remap_records(records, title_column: str):
        remapped = []
        for row in records:
            video_id = str(row.get("videoID") or "")
            combined = dict(row)
            neutral_row = neutralized_rows.get(video_id)
            if neutral_row:
                combined[title_column] = neutral_row.get(title_column, neutral_row.get("neutral_title", ""))
            remapped.append(combined)
        return remapped

    original_records = split["test"].to_pandas().to_dict(orient="records")
    neutral_records = remap_records(original_records, args.title_column_neutralized)

    original_loader = make_loader(original_records, args.batch_size, args.image_size, args.num_workers, train=False)
    neutral_loader = make_loader(neutral_records, args.batch_size, args.image_size, args.num_workers, train=False, title_column=args.title_column_neutralized)

    tokenizer, text_model = load_text_artifacts(args.text_base_model, args.text_checkpoint, device)
    image_model = load_image_artifact(args.image_checkpoint, device)
    fusion_head = load_fusion_head(args.fusion_checkpoint, device)
    sanity_check_models(text_model, tokenizer, image_model, original_loader, device)

    original_fused, labels = extract_logits(original_loader, tokenizer, text_model, image_model, device)
    neutral_fused, _ = extract_logits(neutral_loader, tokenizer, text_model, image_model, device)

    def _head_output_logits(features: np.ndarray) -> np.ndarray:
        """Run fusion head and return [N, 2] logits for shift analysis."""
        fusion_head.eval()
        with torch.no_grad():
            return fusion_head(torch.tensor(features, dtype=torch.float32).to(device)).cpu().numpy()

    original_head_logits = _head_output_logits(original_fused)
    neutral_head_logits = _head_output_logits(neutral_fused)

    original_metrics = evaluate_head(fusion_head, torch.tensor(original_fused, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), device)
    neutral_metrics = evaluate_head(fusion_head, torch.tensor(neutral_fused, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), device)
    # Pass [N, 2] head output so softmax correctly gives P(clickbait) at column 1
    shift_summary = summarize_shift(original_head_logits, neutral_head_logits)

    results = {
        "original_metrics": original_metrics,
        "neutralized_metrics": neutral_metrics,
        "shift_summary": shift_summary,
        "samples": int(len(labels)),
    }
    append_metrics_record(
        args.results_path,
        {
            "model": "bias_experiment",
            "stage": "neutralization_shift",
            "metrics": results,
        },
    )
    print(results)


if __name__ == "__main__":
    main()