from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from clickbait_common import chat_completion, ensure_parent, load_csv_dataset, parse_json_object, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite titles into neutral factual language using OpenRouter.")
    parser.add_argument("--csv", type=Path, default=Path("thumbnails & titles") / "prepared_dataset_titles.csv")
    parser.add_argument("--output", type=Path, default=Path("thumbnails & titles") / "neutralized_titles.csv")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--referer", type=str, default="")
    parser.add_argument("--app-name", type=str, default="clickbait-neutralizer")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def rewrite_title(*, base_url: str, api_key: str, model: str, title: str, referer: str, app_name: str, max_tokens: int) -> str:
    system_prompt = (
        "You rewrite YouTube video titles into neutral factual language. "
        "Preserve the topic and core meaning. Return JSON with keys 'neutral_title' and 'notes'."
    )
    user_prompt = f"Original title: {title}\n\nRewrite it neutrally."
    raw = chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_headers={k: v for k, v in {"HTTP-Referer": referer or None, "X-Title": app_name or None}.items() if v},
    )
    try:
        payload = parse_json_object(raw)
        neutral_title = str(payload.get("neutral_title") or payload.get("title") or "").strip()
        if neutral_title:
            return neutral_title
    except Exception:
        pass
    return raw.strip().strip('"')


def main() -> None:
    args = build_parser().parse_args()
    csv_path = resolve_project_path(args.csv)
    output_path = resolve_project_path(args.output)
    ensure_parent(output_path)

    dataset = load_csv_dataset(csv_path)
    rows = dataset.to_pandas().to_dict(orient="records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["videoID", "is_clickbait", "title", "neutral_title"])
        writer.writeheader()

        for row in tqdm(rows, desc="Neutralizing titles"):
            title = str(row.get("title") or "").strip()
            neutral_title = rewrite_title(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                title=title,
                referer=args.referer,
                app_name=args.app_name,
                max_tokens=args.max_tokens,
            )
            writer.writerow(
                {
                    "videoID": row.get("videoID", ""),
                    "is_clickbait": row.get("is_clickbait", ""),
                    "title": title,
                    "neutral_title": neutral_title,
                }
            )

    print(f"Saved neutralized titles to {output_path}")


if __name__ == "__main__":
    main()