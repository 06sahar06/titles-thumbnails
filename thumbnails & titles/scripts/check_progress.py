import csv
from pathlib import Path

csv_path = Path('non-clickbait.csv')
ids = set()
with csv_path.open('r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = str(row.get('videoID', '')).strip()
        if vid and vid.lower() != 'nan':
            ids.add(vid)

num_ids = len(ids)

thumb_dir = Path('non_clickbait_thumbnails')
count_files = 0
if thumb_dir.exists():
    for p in thumb_dir.iterdir():
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            count_files += 1

missing = num_ids - count_files
print(num_ids)
print(count_files)
print(missing)
