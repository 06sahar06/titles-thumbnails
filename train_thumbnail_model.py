from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from clickbait_common import CHECKPOINTS_DIR, RESULTS_DIR, append_metrics_record, compute_binary_metrics, load_csv_dataset, sanitize_name, stratified_split
from clickbait_common import resolve_project_path
from clickbait_vision import ThumbnailDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 for thumbnail clickbait detection.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_thumbnails.csv")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINTS_DIR / "resnet50_thumbnail")
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


def build_model(freeze_backbone: bool) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def make_loaders(split, batch_size: int, image_size: int, num_workers: int):
    train_dataset = ThumbnailDataset(split["train"].to_pandas().to_dict(orient="records"), image_size=image_size, train=True)
    validation_dataset = ThumbnailDataset(split["validation"].to_pandas().to_dict(orient="records"), image_size=image_size, train=False)
    test_dataset = ThumbnailDataset(split["test"].to_pandas().to_dict(orient="records"), image_size=image_size, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, validation_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    all_logits = []
    all_labels = []
    running_loss = 0.0

    progress = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in progress:
        pixels = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(pixels)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        progress.set_postfix(loss=float(loss.item()))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(logits, labels)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def sanity_check(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    sample_batch = next(iter(loader))
    sample_pixels = sample_batch["pixel_values"][:1].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(sample_pixels)
    print(f"Sanity check logits shape: {tuple(outputs.shape)}")


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = resolve_project_path(args.csv)
    dataset = load_csv_dataset(csv_path)
    split = stratified_split(dataset, seed=args.seed)
    train_loader, validation_loader, test_loader = make_loaders(split, args.batch_size, args.image_size, args.num_workers)

    model = build_model(args.freeze_backbone).to(device)
    sanity_check(model, train_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = resolve_project_path(args.output_dir)
    results_path = resolve_project_path(args.results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        validation_metrics = run_epoch(model, validation_loader, criterion, optimizer, device, train=False)
        append_metrics_record(
            results_path,
            {
                "model": "resnet50_thumbnail",
                "stage": "validation",
                "epoch": epoch,
                "metrics": {"train": train_metrics, "validation": validation_metrics},
            },
        )
        print(f"Epoch {epoch} train: {train_metrics}")
        print(f"Epoch {epoch} validation: {validation_metrics}")

        if validation_metrics["f1_macro"] > best_f1:
            best_f1 = validation_metrics["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "args": vars(args),
                },
                best_path,
            )

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    append_metrics_record(
        results_path,
        {
            "model": "resnet50_thumbnail",
            "stage": "test",
            "metrics": test_metrics,
        },
    )
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()