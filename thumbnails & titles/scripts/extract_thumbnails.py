from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "non-clickbait.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "non_clickbait_thumbnails"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / ".extract_thumbnails_checkpoint.json"


def sanitize_video_id(video_id: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(video_id))


def load_checkpoint(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0

    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return int(data.get("next_row", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def save_checkpoint(checkpoint_path: Path, next_row: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps({"next_row": next_row}), encoding="utf-8")


def download_thumbnail(video_id: str) -> bytes:
    candidate_urls = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/default.jpg",
    ]

    for url in candidate_urls:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read()
                if data:
                    return data
        except (HTTPError, URLError, TimeoutError, OSError):
            continue

    return b""


def process_row(video_id: str, output_dir: Path) -> tuple[str, bool]:
    target_file = output_dir / f"{sanitize_video_id(video_id)}.jpg"
    if target_file.exists():
        return video_id, True

    thumbnail_bytes = download_thumbnail(video_id)
    if not thumbnail_bytes:
        return video_id, False

    try:
        target_file.write_bytes(thumbnail_bytes)
        return video_id, True
    except OSError:
        return video_id, False


def extract_thumbnails(csv_path: Path, output_dir: Path, checkpoint_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    failures = 0
    start_row = load_checkpoint(checkpoint_path)

    rows: list[tuple[int, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if row_number <= start_row:
                continue

            video_id = str(row.get("videoID", "")).strip()
            if not video_id or video_id.lower() == "nan":
                failures += 1
                save_checkpoint(checkpoint_path, row_number)
                continue

            rows.append((row_number, video_id))

    if rows:
        with ThreadPoolExecutor(max_workers=24) as executor:
            future_to_row = {
                executor.submit(process_row, video_id, output_dir): row_number
                for row_number, video_id in rows
            }
            completed_rows = 0
            for future in as_completed(future_to_row):
                row_number = future_to_row[future]
                try:
                    _, succeeded = future.result()
                    if succeeded:
                        frame_count += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
                completed_rows += 1
                save_checkpoint(checkpoint_path, row_number)

            if completed_rows > 0:
                save_checkpoint(checkpoint_path, start_row + completed_rows)

    print(f"Saved {frame_count} thumbnails to {output_dir}")
    print(f"Skipped or failed on {failures} rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract thumbnails for video IDs in non-clickbait.csv.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the source CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write thumbnail images.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to the resume checkpoint file.")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    extract_thumbnails(arguments.csv, arguments.output, arguments.checkpoint)


if __name__ == "__main__":
    main()
