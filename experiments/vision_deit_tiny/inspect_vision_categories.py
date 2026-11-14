
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/raw/VISION")

if not ROOT.exists():
    raise SystemExit(f"Dataset folder not found: {ROOT.resolve()}")

categories = ["flat", "nat", "natFBH", "natFBL", "natWA"]
counts = defaultdict(lambda: defaultdict(int))

for device_folder in ROOT.iterdir():
    if not device_folder.is_dir():
        continue

    images_folder = device_folder / "images"

    if not images_folder.exists():
        continue

    for category in categories:
        category_folder = images_folder / category

        if category_folder.exists():
            count = sum(
                1 for p in category_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in
                {".jpg", ".jpeg", ".png"}
            )
            counts[device_folder.name][category] = count

print("\n========== VISION CATEGORY SUMMARY ==========\n")

totals = defaultdict(int)

header = f"{'DEVICE':<32}"
for category in categories:
    header += f"{category:>10}"
print(header)
print("-" * len(header))

for device, category_counts in sorted(counts.items()):
    row = f"{device:<32}"

    for category in categories:
        value = category_counts[category]
        totals[category] += value
        row += f"{value:>10}"

    print(row)

print("-" * len(header))

row = f"{'TOTAL':<32}"
for category in categories:
    row += f"{totals[category]:>10}"
print(row)

print("\nTotal devices:", len(counts))
print("Total images:", sum(totals.values()))

print("\n========== SUMMARY COMPLETE ==========")
