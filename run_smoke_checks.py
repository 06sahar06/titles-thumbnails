from __future__ import annotations

import sys
from pathlib import Path

def main():
    print("Smoke checks for Clickbait project")
    print(f"Python: {sys.version.splitlines()[0]}")

    try:
        import torch
        import transformers
        import datasets
        import peft
        import torchvision
        import PIL
        import sklearn
        import requests
        import numpy as np
        print(f"torch: {torch.__version__} (cuda available: {torch.cuda.is_available()})")
        print(f"transformers: {transformers.__version__}")
        print(f"datasets: {datasets.__version__}")
        print(f"peft: {peft.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        print(f"Pillow: {PIL.__version__}")
        print(f"scikit-learn: {sklearn.__version__}")
        print(f"requests: {requests.__version__}")
        print(f"numpy: {np.__version__}")
    except Exception as e:
        print("Import check failed:", e)
        raise

    root = Path(__file__).resolve().parent
    csv_titles = root / "thumbnails & titles" / "prepared_dataset_titles.csv"
    csv_thumbs = root / "thumbnails & titles" / "prepared_dataset_thumbnails.csv"
    thumb_dirs = [root / "thumbnails & titles" / "clickbait_thumbnails", root / "thumbnails & titles" / "non_clickbait_thumbnails"]

    print(f"Checking CSVs: {csv_titles.exists()} {csv_thumbs.exists()}")
    for d in thumb_dirs:
        if d.exists():
            count = sum(1 for _ in d.glob("**/*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            print(f"{d}: exists, image count = {count}")
        else:
            print(f"{d}: MISSING")

    print("Smoke checks completed.")

if __name__ == '__main__':
    main()
