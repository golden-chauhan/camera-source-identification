
from pathlib import Path
from collections import Counter
from PIL import Image, ExifTags
import statistics

IMAGE_ROOT = Path("data/raw/VISION")

if not IMAGE_ROOT.exists():
    print(f"ERROR: Folder not found: {IMAGE_ROOT.resolve()}")
    raise SystemExit(1)

extensions = Counter()
dimensions = Counter()
folder_counts = Counter()
exif_count = 0
jpeg_count = 0
corrupt_count = 0
file_sizes = []
samples = []

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

for path in IMAGE_ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in image_extensions:
        continue

    extensions[path.suffix.lower()] += 1
    folder_counts[str(path.parent.relative_to(IMAGE_ROOT))] += 1
    file_sizes.append(path.stat().st_size)

    try:
        with Image.open(path) as img:
            dimensions[img.size] += 1

            if img.format == "JPEG":
                jpeg_count += 1

            exif = img.getexif()
            if exif:
                exif_count += 1

            if len(samples) < 15:
                samples.append({
                    "path": str(path.relative_to(IMAGE_ROOT)),
                    "format": img.format,
                    "dimensions": img.size,
                    "mode": img.mode,
                    "exif_fields": len(exif)
                })

    except Exception as error:
        corrupt_count += 1
        print(f"Could not read: {path} | {error}")

total = sum(extensions.values())

print("\n========== IMAGE DATASET INSPECTION ==========")
print(f"Dataset folder: {IMAGE_ROOT.resolve()}")
print(f"Total images: {total}")
print(f"JPEG images: {jpeg_count}")
print(f"Images containing EXIF: {exif_count}")
print(f"Unreadable images: {corrupt_count}")

print("\n--- File formats ---")
for ext, count in extensions.most_common():
    print(f"{ext}: {count}")

print("\n--- Most common image dimensions ---")
for size, count in dimensions.most_common(20):
    print(f"{size[0]} x {size[1]}: {count} images")

print("\n--- Folder structure (first 30 folders) ---")
for folder, count in folder_counts.most_common(30):
    print(f"{folder}: {count} images")

print("\n--- Sample images ---")
for item in samples:
    print(item)

if file_sizes:
    print("\n--- File sizes ---")
    print(f"Smallest: {min(file_sizes) / 1024:.1f} KB")
    print(f"Largest: {max(file_sizes) / 1024:.1f} KB")
    print(f"Average: {statistics.mean(file_sizes) / 1024:.1f} KB")

print("\n========== INSPECTION COMPLETE ==========")
