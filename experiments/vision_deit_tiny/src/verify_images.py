
from pathlib import Path
import csv
import shutil

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]

METADATA = PROJECT_DIR / "data" / "processed" / "vision_metadata.csv"
BACKUP = PROJECT_DIR / "data" / "processed" / "vision_metadata_backup.csv"
REPORT = PROJECT_DIR / "outputs" / "image_verification_report.csv"

REPORT.parent.mkdir(parents=True, exist_ok=True)

with METADATA.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("Metadata file is empty.")

valid_rows = []
bad_rows = []

print(f"Checking {len(rows)} images...")

for index, row in enumerate(rows, start=1):
    image_path = PROJECT_DIR / row["filepath"]

    try:
        with Image.open(image_path) as image:
            image.verify()

        # Reopen and fully decode the image.
        with Image.open(image_path) as image:
            image.convert("RGB").load()

        valid_rows.append(row)

    except Exception as error:
        bad_rows.append({
            "filepath": row["filepath"],
            "error": str(error)
        })

    if index % 250 == 0 or index == len(rows):
        print(f"Checked {index}/{len(rows)}")

# Preserve original metadata before updating it.
shutil.copy2(METADATA, BACKUP)

# Rewrite metadata using only verified images.
with METADATA.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(valid_rows)

# Save a report of rejected images.
with REPORT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filepath", "error"]
    )
    writer.writeheader()
    writer.writerows(bad_rows)

print("\n========== VERIFICATION SUMMARY ==========")
print("Original images:", len(rows))
print("Valid images:", len(valid_rows))
print("Corrupted images:", len(bad_rows))
print("Backup:", BACKUP)
print("Report:", REPORT)

if not valid_rows:
    raise RuntimeError("No valid images remain.")
