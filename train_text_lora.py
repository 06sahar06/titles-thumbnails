from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from clickbait_common import (
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    SPLIT_INDICES_PATH,
    append_metrics_record,
    compute_binary_metrics,
    load_csv_dataset,
    resolve_project_path,
    sanitize_name,
    save_split_indices,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Weighted loss trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with LoRA for clickbait detection.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_titles.csv")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINTS_DIR / "distilbert_lora")
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--split-indices-path", type=Path, default=SPLIT_INDICES_PATH,
                        help="Where to save the canonical train/val/test videoID split.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--undersample-ratio", type=float, default=2.0,
                        help="Clickbait:non-clickbait ratio after undersampling. 0 = no undersampling.")
    return parser


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class MetricsSaverCallback(TrainerCallback):
    def __init__(self, results_path: Path, model_name: str) -> None:
        self.results_path = results_path
        self.model_name = model_name

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        append_metrics_record(
            self.results_path,
            {
                "model": self.model_name,
                "stage": "validation",
                "step": int(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "metrics": metrics,
            },
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def undersample_dataset(dataset, label_column, majority_label, ratio, seed):
    labels = np.array(dataset[label_column])
    minority_indices = np.where(labels != majority_label)[0].tolist()
    majority_indices = np.where(labels == majority_label)[0].tolist()
    target_majority = int(len(minority_indices) * ratio)
    rng = np.random.default_rng(seed)
    sampled_majority = rng.choice(majority_indices, size=target_majority, replace=False).tolist()
    combined = sorted(minority_indices + sampled_majority)
    result = dataset.select(combined).shuffle(seed=seed)
    print(
        f"Undersampled training set: {len(minority_indices)} non-clickbait + "
        f"{target_majority} clickbait = {len(combined)} total (ratio 1:{ratio:.1f})"
    )
    return result


def compute_class_weights(dataset, label_column, device):
    df = dataset.to_pandas()
    counts = df[label_column].value_counts().sort_index()
    total = len(df)
    weights = [total / (2.0 * counts[i]) for i in sorted(counts.index)]
    print(f"Class counts: {dict(counts)}")
    print(f"Class weights: non-clickbait={weights[0]:.4f}, clickbait={weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def tokenize_dataset(dataset, tokenizer, max_length):
    def _tokenize(batch):
        encoded = tokenizer(batch["title"], truncation=True, max_length=max_length)
        encoded["labels"] = batch["is_clickbait"]
        return encoded
    remove_columns = list(dataset["train"].column_names)
    return dataset.map(_tokenize, batched=True, remove_columns=remove_columns)


# ---------------------------------------------------------------------------
# Metrics / sanity check
# ---------------------------------------------------------------------------

def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return compute_binary_metrics(np.asarray(predictions), np.asarray(eval_prediction.label_ids))


def sanity_check(model, tokenizer, device):
    model.eval()
    sample = tokenizer("A neutral sample title for sanity checking.", return_tensors="pt", truncation=True, padding=True)
    sample = {k: v.to(device) for k, v in sample.items()}
    with torch.no_grad():
        outputs = model(**sample)
    print(f"Sanity check logits shape: {tuple(outputs.logits.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split
    csv_path = resolve_project_path(args.csv)
    dataset = load_csv_dataset(csv_path)
    split = stratified_split(dataset, seed=args.seed)

    # Save canonical split indices (first run only — all other scripts load from this file)
    split_indices_path = resolve_project_path(args.split_indices_path)
    if not split_indices_path.exists():
        save_split_indices(split, split_indices_path)
    else:
        print(f"Split indices already exist at {split_indices_path} — skipping save.")

    # Undersample training split only
    if args.undersample_ratio > 0:
        train_dataset = undersample_dataset(
            split["train"], "is_clickbait", majority_label=1,
            ratio=args.undersample_ratio, seed=args.seed,
        )
    else:
        train_dataset = split["train"]
        print("No undersampling applied.")

    split = DatasetDict(train=train_dataset, validation=split["validation"], test=split["test"])

    class_weights = compute_class_weights(split["train"], "is_clickbait", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    tokenized = tokenize_dataset(split, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={0: "non_clickbait", 1: "clickbait"},
        label2id={"non_clickbait": 0, "clickbait": 1},
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    sanity_check(model, tokenizer, device)

    output_dir = resolve_project_path(args.output_dir)
    results_path = resolve_project_path(args.results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_kwargs = dict(
        output_dir=str(output_dir), learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, logging_strategy="steps", logging_steps=25,
        report_to=[], save_total_limit=2, remove_unused_columns=False,
    )
    try:
        training_arguments = TrainingArguments(eval_strategy="epoch", **shared_kwargs)
    except TypeError:
        try:
            training_arguments = TrainingArguments(evaluation_strategy="epoch", **shared_kwargs)
        except TypeError:
            print("Warning: TrainingArguments fallback to minimal config.")
            training_arguments = TrainingArguments(
                output_dir=str(output_dir), num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                max_grad_norm=args.max_grad_norm, logging_steps=25,
            )

    from inspect import signature
    trainer_init_params = signature(Trainer.__init__).parameters
    trainer_kwargs: dict = {
        "model": model, "args": training_arguments,
        "train_dataset": tokenized["train"], "eval_dataset": tokenized["validation"],
        "compute_metrics": compute_metrics,
        "callbacks": [MetricsSaverCallback(results_path, sanitize_name(args.model_name))],
        "class_weights": class_weights,
    }
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "data_collator" in trainer_init_params:
        trainer_kwargs["data_collator"] = DataCollatorWithPadding(tokenizer=tokenizer)

    try:
        trainer = WeightedTrainer(**trainer_kwargs)
    except TypeError:
        print("Warning: WeightedTrainer init failed; retrying with minimal kwargs.")
        trainer = WeightedTrainer(**{k: trainer_kwargs[k] for k in (
            "model", "args", "train_dataset", "eval_dataset", "class_weights")})

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_output = trainer.predict(tokenized["test"])
    test_logits = np.asarray(
        test_output.predictions[0] if isinstance(test_output.predictions, tuple)
        else test_output.predictions
    )
    test_metrics = compute_binary_metrics(test_logits, np.asarray(test_output.label_ids))
    append_metrics_record(results_path, {
        "model": sanitize_name(args.model_name), "stage": "test", "metrics": test_metrics,
    })
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
