from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clickbait_common import CHECKPOINTS_DIR, RESULTS_DIR, append_metrics_record, compute_binary_metrics, load_csv_dataset, resolve_project_path, sanitize_name, stratified_split
from clickbait_vision import ThumbnailDataset


class FusionSampleDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_size: int = 224, train: bool = False, title_column: str = "title") -> None:
        self.records = records
        self.title_column = title_column
        self.thumbnail_dataset = ThumbnailDataset(records, image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image_record = self.thumbnail_dataset[index]
        image_record["title"] = str(row.get(self.title_column) or "")
        return image_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Late-fusion clickbait detector over text and thumbnail logits.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_thumbnails.csv")
    parser.add_argument("--text-base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--text-checkpoint", type=Path, default=CHECKPOINTS_DIR / "distilbert_lora")
    parser.add_argument("--image-checkpoint", type=Path, default=CHECKPOINTS_DIR / "resnet50_thumbnail" / "best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINTS_DIR / "multimodal_fusion")
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def load_text_artifacts(text_base_model: str, text_checkpoint: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(text_checkpoint if text_checkpoint.exists() else text_base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        text_base_model,
        num_labels=2,
        id2label={0: "non_clickbait", 1: "clickbait"},
        label2id={"non_clickbait": 0, "clickbait": 1},
    )
    model = PeftModel.from_pretrained(base_model, text_checkpoint)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_image_artifact(image_checkpoint: Path, device: torch.device):
    from torchvision.models import ResNet50_Weights, resnet50

    checkpoint = torch.load(image_checkpoint, map_location=device)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def sanity_check_models(text_model, tokenizer, image_model, image_loader, device: torch.device) -> None:
    sample_text = tokenizer("This is a sanity-check title.", return_tensors="pt", truncation=True, padding=True)
    sample_text = {key: value.to(device) for key, value in sample_text.items()}
    sample_image = next(iter(image_loader))["pixel_values"][:1].to(device)
    with torch.no_grad():
        text_logits = text_model(**sample_text).logits
        image_logits = image_model(sample_image)
    print(f"Sanity check text logits shape: {tuple(text_logits.shape)}")
    print(f"Sanity check image logits shape: {tuple(image_logits.shape)}")


def extract_logits(dataset_loader: DataLoader, tokenizer, text_model, image_model, device: torch.device):
    all_logits = []
    all_labels = []

    for batch in tqdm(dataset_loader, desc="Extracting fusion features", leave=False):
        titles = list(batch["title"])
        labels = batch["labels"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        encoded = tokenizer(titles, truncation=True, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            text_logits = text_model(**encoded).logits
            image_logits = image_model(pixel_values)
        fused_logits = torch.cat([text_logits, image_logits], dim=1)
        all_logits.append(fused_logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def train_fusion_head(train_features: torch.Tensor, train_labels: torch.Tensor, validation_features: torch.Tensor, validation_labels: torch.Tensor, epochs: int, lr: float, device: torch.device):
    head = nn.Linear(train_features.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        head.train()
        for features, labels in tqdm(train_loader, desc=f"Fusion head epoch {epoch}", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_metrics = evaluate_head(head, validation_features, validation_labels, device)
        print(f"Fusion epoch {epoch} validation: {validation_metrics}")
        if validation_metrics["f1_macro"] > best_f1:
            best_f1 = validation_metrics["f1_macro"]
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def evaluate_head(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, float]:
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device)).cpu().numpy()
    return compute_binary_metrics(logits, labels.cpu().numpy())


def make_loader(records: list[dict[str, Any]], batch_size: int, image_size: int, num_workers: int, train: bool, title_column: str = "title") -> DataLoader:
    dataset = FusionSampleDataset(records, image_size=image_size, train=train, title_column=title_column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = resolve_project_path(args.csv)
    dataset = load_csv_dataset(csv_path)
    split = stratified_split(dataset, seed=args.seed)
    train_records = split["train"].to_pandas().to_dict(orient="records")
    validation_records = split["validation"].to_pandas().to_dict(orient="records")
    test_records = split["test"].to_pandas().to_dict(orient="records")

    train_loader = make_loader(train_records, args.batch_size, args.image_size, args.num_workers, train=True)
    validation_loader = make_loader(validation_records, args.batch_size, args.image_size, args.num_workers, train=False)
    test_loader = make_loader(test_records, args.batch_size, args.image_size, args.num_workers, train=False)

    tokenizer, text_model = load_text_artifacts(args.text_base_model, args.text_checkpoint, device)
    image_model = load_image_artifact(args.image_checkpoint, device)
    sanity_check_models(text_model, tokenizer, image_model, train_loader, device)

    train_logits, train_labels = extract_logits(train_loader, tokenizer, text_model, image_model, device)
    validation_logits, validation_labels = extract_logits(validation_loader, tokenizer, text_model, image_model, device)
    test_logits, test_labels = extract_logits(test_loader, tokenizer, text_model, image_model, device)

    train_features = torch.tensor(train_logits, dtype=torch.float32)
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    validation_features = torch.tensor(validation_logits, dtype=torch.float32)
    validation_targets = torch.tensor(validation_labels, dtype=torch.long)
    test_features = torch.tensor(test_logits, dtype=torch.float32)
    test_targets = torch.tensor(test_labels, dtype=torch.long)

    fusion_head = train_fusion_head(train_features, train_targets, validation_features, validation_targets, args.epochs, args.lr, device)

    output_dir = resolve_project_path(args.output_dir)
    results_path = resolve_project_path(args.results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fusion_head.pt"
    torch.save(
        {
            "model_state_dict": fusion_head.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )

    test_metrics = evaluate_head(fusion_head, test_features, test_targets, device)
    append_metrics_record(
        results_path,
        {
            "model": "multimodal_fusion",
            "stage": "test",
            "metrics": test_metrics,
        },
    )
    print(f"Fusion test metrics: {test_metrics}")
    print(f"Saved fusion head to {checkpoint_path}")


if __name__ == "__main__":
    main()