
from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "vision_metadata.csv"
)


def read_metadata():
    with METADATA_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def main():
    rows = read_metadata()

    category_counts = defaultdict(Counter)
    device_category_counts = defaultdict(Counter)
    split_device_counts = defaultdict(Counter)

    for row in rows:
        split = row["split"]
        device = row["device_id"]

        parts = row["filepath"].replace("\\", "/").split("/")

        # Expected path:
        # data/raw/VISION/DEVICE/images/CATEGORY/image.jpg
        if len(parts) < 2:
            raise ValueError(
                f"Unexpected image path: {row['filepath']}"
            )

        category = parts[-2]

        category_counts[split][category] += 1
        device_category_counts[(split, device)][category] += 1
        split_device_counts[split][device] += 1

    print("\n========== CATEGORY DISTRIBUTION ==========")

    all_categories = sorted({
        category
        for counts in category_counts.values()
        for category in counts
    })

    for split in ("train", "validation", "test"):
        total = sum(category_counts[split].values())

        print(f"\n{split.upper()} ({total} images)")

        for category in all_categories:
            count = category_counts[split][category]
            percentage = 100 * count / total if total else 0

            print(
                f"{category:12s} "
                f"{count:5d} images "
                f"({percentage:6.2f}%)"
            )

    print("\n========== PER-DEVICE CATEGORY DISTRIBUTION ==========")

    devices = sorted({
        row["device_id"] for row in rows
    })

    for device in devices:
        print(f"\n{device}")

        for split in ("train", "validation", "test"):
            counts = device_category_counts[(split, device)]

            print(
                f"  {split:10s}: "
                + ", ".join(
                    f"{category}={count}"
                    for category, count in sorted(counts.items())
                )
            )

    print("\n========== SPLIT TOTALS ==========")

    for split in ("train", "validation", "test"):
        print(
            f"{split}: "
            f"{sum(split_device_counts[split].values())}"
        )


if __name__ == "__main__":
    main()
