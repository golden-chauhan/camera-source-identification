
from pathlib import Path
import csv
import shutil
from collections import Counter

from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

METADATA_PATH = PROCESSED_DIR / "vision_metadata.csv"
MAPPING_PATH = PROCESSED_DIR / "device_label_mapping.csv"

BACKUP_PATH = (
    PROCESSED_DIR /
    "vision_metadata_before_regularized_split.csv"
)

RANDOM_STATE = 42
TEST_IMAGES = 500
VALIDATION_IMAGES = 676


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(MAPPING_PATH)

    rows = read_csv(METADATA_PATH)
    mapping = read_csv(MAPPING_PATH)

    if not rows:
        raise ValueError("Metadata contains no images.")

    if not BACKUP_PATH.exists():
        shutil.copy2(METADATA_PATH, BACKUP_PATH)

    # Remove previous split assignments.
    for row in rows:
        row.pop("split", None)

    # Check unique image paths.
    paths = [row["filepath"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths detected.")

    # Validate class mapping.
    mapping.sort(key=lambda row: int(row["device_label"]))

    class_names = [
        row["device_id"] for row in mapping
    ]

    expected_labels = set(range(len(class_names)))

    labels = [
        int(row["device_label"]) for row in rows
    ]

    if set(labels) != expected_labels:
        raise ValueError(
            "Metadata labels do not match the device mapping."
        )

    if len(class_names) != 22:
        raise ValueError(
            f"Expected 22 device classes, found {len(class_names)}."
        )

    counts = Counter(labels)

    insufficient = {
        label: count
        for label, count in counts.items()
        if count < 3
    }

    if insufficient:
        raise ValueError(
            f"Some classes have too few images for stratification: "
            f"{insufficient}"
        )

    total = len(rows)

    if total <= TEST_IMAGES + VALIDATION_IMAGES:
        raise ValueError("Not enough images for requested split.")

    print("Verified images available:", total)

    # Reserve exactly 500 images for testing.
    train_val, test_rows = train_test_split(
        rows,
        test_size=TEST_IMAGES,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Reserve exactly 676 images for validation.
    train_val_labels = [
        int(row["device_label"]) for row in train_val
    ]

    train_rows, validation_rows = train_test_split(
        train_val,
        test_size=VALIDATION_IMAGES,
        random_state=RANDOM_STATE,
        stratify=train_val_labels
    )

    for row in train_rows:
        row["split"] = "train"

    for row in validation_rows:
        row["split"] = "validation"

    for row in test_rows:
        row["split"] = "test"

    final_rows = train_rows + validation_rows + test_rows

    # Confirm exact split sizes.
    assert len(train_rows) == total - TEST_IMAGES - VALIDATION_IMAGES
    assert len(validation_rows) == VALIDATION_IMAGES
    assert len(test_rows) == TEST_IMAGES

    # Confirm no overlap.
    split_paths = {}

    for split in ("train", "validation", "test"):
        split_paths[split] = {
            row["filepath"]
            for row in final_rows
            if row["split"] == split
        }

    assert split_paths["train"].isdisjoint(
        split_paths["validation"]
    )

    assert split_paths["train"].isdisjoint(
        split_paths["test"]
    )

    assert split_paths["validation"].isdisjoint(
        split_paths["test"]
    )

    fieldnames = [
        "filepath",
        "device_id",
        "camera_model",
        "device_label",
        "split"
    ]

    write_csv(
        METADATA_PATH,
        final_rows,
        fieldnames
    )

    print("\n========== REGULARIZED DATASET SPLIT ==========")
    print("Total images:", total)
    print("Device classes:", len(class_names))
    print("Training images:", len(train_rows))
    print("Validation images:", len(validation_rows))
    print("Test images:", len(test_rows))

    print("\nImages per device:")

    for label in sorted(counts):
        device_name = class_names[label]

        device_rows = [
            row for row in final_rows
            if int(row["device_label"]) == label
        ]

        split_counts = Counter(
            row["split"] for row in device_rows
        )

        print(
            f"{device_name}: "
            f"train={split_counts['train']}, "
            f"validation={split_counts['validation']}, "
            f"test={split_counts['test']}"
        )

    print("\nAll integrity checks passed.")
    print("Metadata:", METADATA_PATH)
    print("Backup:", BACKUP_PATH)


if __name__ == "__main__":
    main()
