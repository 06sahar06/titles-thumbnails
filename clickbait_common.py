from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "thumbnails & titles"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
THUMBNAIL_DIRS = [DATA_DIR / "clickbait_thumbnails", DATA_DIR / "non_clickbait_thumbnails"]


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "model"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def load_csv_dataset(csv_path: Path) -> Dataset:
    loaded = load_dataset("csv", data_files={"data": str(csv_path.resolve())}, split="data")
    return loaded


def stratified_split(
    dataset: Dataset,
    label_column: str = "is_clickbait",
    seed: int = 42,
) -> DatasetDict:
    labels = np.asarray(dataset[label_column], dtype=int)
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = labels[temp_indices]
    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    return DatasetDict(
        train=dataset.select(sorted(train_indices.tolist())),
        validation=dataset.select(sorted(validation_indices.tolist())),
        test=dataset.select(sorted(test_indices.tolist())),
    )


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = _softmax(logits)
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def append_metrics_record(metrics_path: Path, record: dict[str, Any]) -> None:
    ensure_parent(metrics_path)
    existing: dict[str, Any]
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    history = existing.get("history")
    if not isinstance(history, list):
        history = []

    history.append(_json_safe({**record, "timestamp_utc": datetime.now(timezone.utc).isoformat()}))

    payload = {
        **existing,
        "history": history,
    }
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def resolve_thumbnail_path(video_id: str, fallback_dirs: list[Path] | None = None) -> Path | None:
    candidate_dirs = fallback_dirs or THUMBNAIL_DIRS
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    for directory in candidate_dirs:
        for extension in extensions:
            candidate = directory / f"{video_id}{extension}"
            if candidate.exists():
                return candidate

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for extension in extensions:
            matches = sorted(directory.glob(f"{video_id}*{extension}"))
            if matches:
                return matches[0]

    return None


def image_to_data_url(image_path: Path) -> str:
    mime = "image/jpeg"
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
    temperature: float = 0.0,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty chat completion response: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        content = "\n".join(parts)
    return str(content).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()