from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "non-clickbait.csv"
DEFAULT_OUTPUT = ROOT_DIR / "non_clickbait_titles.csv"
DEFAULT_CHECKPOINT = ROOT_DIR / "non_clickbait_titles_checkpoint.json"
DEFAULT_API_KEY = ""


def load_processed(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    processed = set()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get("videoID") or "").strip()
            if vid:
                processed.add(vid)
    return processed


def save_checkpoint(path: Path, next_index: int) -> None:
    path.write_text(json.dumps({"next_index": next_index}), encoding="utf-8")


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("next_index", 0))
    except Exception:
        return 0


def fetch_titles_batch(video_ids: list[str], api_key: str) -> dict[str, str]:
    query = urlencode({"part": "snippet", "id": ",".join(video_ids), "key": api_key})
    api_url = f"https://www.googleapis.com/youtube/v3/videos?{query}"

    try:
        with urlopen(api_url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return {}

    titles_by_id: dict[str, str] = {}
    for item in payload.get("items") or []:
        video_id = str(item.get("id") or "").strip()
        snippet = item.get("snippet") or {}
        if video_id:
            titles_by_id[video_id] = str(snippet.get("title") or "")

    return titles_by_id


def chunked(sequence: list[dict], size: int) -> list[list[dict]]:
    return [sequence[index:index + size] for index in range(0, len(sequence), size)]


def fetch_titles(csv_path: Path, output_path: Path, checkpoint_path: Path, api_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed(output_path)
    next_index = load_checkpoint(checkpoint_path)

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)

    with output_path.open("a", encoding="utf-8", newline="") as outf:
        writer = None
        batch_rows = rows[next_index:]
        for batch_start in range(0, len(batch_rows), 50):
            batch = batch_rows[batch_start:batch_start + 50]
            ids_in_batch: list[str] = []
            row_metadata: list[tuple[int, str]] = []
            for offset, row in enumerate(batch):
                idx = next_index + batch_start + offset
                video_id = (row.get("videoID") or "").strip()
                row_metadata.append((idx, video_id))
                if video_id and video_id.lower() != "nan" and video_id not in processed:
                    ids_in_batch.append(video_id)

            titles_by_id = fetch_titles_batch(ids_in_batch, api_key) if ids_in_batch else {}

            for idx, video_id in row_metadata:
                if not video_id or video_id.lower() == "nan":
                    save_checkpoint(checkpoint_path, idx + 1)
                    continue
                if video_id in processed:
                    save_checkpoint(checkpoint_path, idx + 1)
                    continue

                title = titles_by_id.get(video_id, "")

                if writer is None:
                    writer = csv.DictWriter(outf, fieldnames=["videoID", "title"])
                    if outf.tell() == 0:
                        writer.writeheader()

                writer.writerow({"videoID": video_id, "title": title})
                processed.add(video_id)
                save_checkpoint(checkpoint_path, idx + 1)

    print(f"Processed up to {load_checkpoint(checkpoint_path)} / {total}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract YouTube titles for videoIDs using YouTube API")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="YouTube API key")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key:
        print("Error: --api-key is required")
        exit(1)
    fetch_titles(args.csv, args.output, args.checkpoint, args.api_key)


if __name__ == "__main__":
    main()
