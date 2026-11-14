
from pathlib import Path
from collections import defaultdict
import csv
import json

from matching import load_fingerprints, identify_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "vision_metadata.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "prnu" / "outputs"

SPLIT_NAME = "test"


def load_test_metadata():
    """Load only the existing held-out test split."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    test_rows = []

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "filepath",
            "device_id",
            "camera_model",
            "device_label",
            "split"
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing metadata columns: {sorted(missing)}"
            )

        for row in reader:
            if row["split"].strip().lower() != SPLIT_NAME:
                continue

            image_path = PROJECT_ROOT / row["filepath"]

            test_rows.append({
                "filepath": row["filepath"],
                "image_path": image_path,
                "true_device": row["device_id"],
                "camera_model": row["camera_model"],
                "device_label": row["device_label"]
            })

    return test_rows


def calculate_metrics(records):
    """Calculate overall and per-device identification metrics."""
    stats = defaultdict(
        lambda: {
            "total": 0,
            "top1_correct": 0,
            "top3_correct": 0
        }
    )

    for record in records:
        device = record["true_device"]

        stats[device]["total"] += 1
        stats[device]["top1_correct"] += int(
            record["top1_correct"]
        )
        stats[device]["top3_correct"] += int(
            record["top3_correct"]
        )

    total = len(records)

    top1_correct = sum(
        int(record["top1_correct"]) for record in records
    )

    top3_correct = sum(
        int(record["top3_correct"]) for record in records
    )

    per_device = []

    for device, values in sorted(stats.items()):
        count = values["total"]

        per_device.append({
            "device": device,
            "test_images": count,
            "top1_correct": values["top1_correct"],
            "top1_accuracy": values["top1_correct"] / count,
            "top3_correct": values["top3_correct"],
            "top3_accuracy": values["top3_correct"] / count
        })

    macro_top1 = sum(
        item["top1_accuracy"] for item in per_device
    ) / len(per_device) if per_device else 0.0

    return {
        "total": total,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "top3_accuracy": top3_correct / total if total else 0.0,
        "macro_top1_accuracy": macro_top1,
        "per_device": per_device
    }


def save_predictions(records):
    path = OUTPUT_ROOT / "prnu_test_predictions.csv"

    if not records:
        return path

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    return path


def save_per_device_metrics(per_device):
    path = OUTPUT_ROOT / "prnu_per_device_metrics.csv"

    if not per_device:
        return path

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_device[0].keys())
        )
        writer.writeheader()
        writer.writerows(per_device)

    return path


def save_confusion_matrix(records):
    """Save actual-device versus predicted-device counts."""
    devices = sorted({
        record["true_device"] for record in records
    } | {
        record["predicted_device"] for record in records
    })

    matrix = {
        actual: {predicted: 0 for predicted in devices}
        for actual in devices
    }

    for record in records:
        actual = record["true_device"]
        predicted = record["predicted_device"]
        matrix[actual][predicted] += 1

    path = OUTPUT_ROOT / "prnu_confusion_matrix.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual_device"] + devices)

        for actual in devices:
            writer.writerow(
                [actual] + [
                    matrix[actual][predicted]
                    for predicted in devices
                ]
            )

    return path


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== PRNU FINAL TEST EVALUATION ==========")

    test_rows = load_test_metadata()

    if not test_rows:
        raise RuntimeError(
            "No test images found in vision_metadata.csv."
        )

    fingerprints = load_fingerprints()

    print("Fingerprints loaded:", len(fingerprints))
    print("Test images found:", len(test_rows))

    records = []
    failed = 0

    predictions_path = (
        OUTPUT_ROOT / "prnu_test_predictions.csv"
    )

    for index, row in enumerate(test_rows, start=1):
        image_path = row["image_path"]
        true_device = row["true_device"]

        print(
            f"\n[{index}/{len(test_rows)}] "
            f"{true_device} | {image_path.name}"
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            results, skipped = identify_image(
                image_path,
                fingerprints
            )

            if not results:
                raise RuntimeError(
                    "No compatible device fingerprints."
                )

            ranked_devices = [
                item["device"] for item in results
            ]

            predicted_device = ranked_devices[0]

            top1_correct = (
                predicted_device == true_device
            )

            top3_correct = (
                true_device in ranked_devices[:3]
            )

            record = {
                "true_device": true_device,
                "camera_model": row["camera_model"],
                "device_label": row["device_label"],
                "predicted_device": predicted_device,
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "top1_pce": results[0]["pce"],
                "top1_ncc": results[0]["ncc"],
                "compatible_devices": len(results),
                "skipped_devices": len(skipped),
                "image_path": row["filepath"]
            }

            records.append(record)

            print(
                "Prediction:",
                predicted_device,
                "| Top-1:",
                "CORRECT" if top1_correct else "WRONG"
            )

        except Exception as error:
            failed += 1

            print("FAILED:", error)

        # Save progress after each image.
        save_predictions(records)

        if index % 10 == 0:
            print(
                f"\nProgress saved: {len(records)} successful images"
            )

    if not records:
        raise RuntimeError(
            "No test images were successfully evaluated."
        )

    metrics = calculate_metrics(records)

    per_device_path = save_per_device_metrics(
        metrics["per_device"]
    )

    confusion_path = save_confusion_matrix(records)

    summary = {
        "evaluation": "PRNU existing held-out test split",
        "split": SPLIT_NAME,
        "fingerprints": len(fingerprints),
        "test_images_requested": len(test_rows),
        "successful_images": metrics["total"],
        "failed_images": failed,
        "top1_correct": metrics["top1_correct"],
        "top1_accuracy": metrics["top1_accuracy"],
        "top3_correct": metrics["top3_correct"],
        "top3_accuracy": metrics["top3_accuracy"],
        "macro_top1_accuracy": metrics["macro_top1_accuracy"],
        "devices_evaluated": len(metrics["per_device"])
    }

    summary_path = (
        OUTPUT_ROOT / "prnu_final_test_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== FINAL PRNU TEST RESULTS ==========")

    print("Test images:", len(test_rows))
    print("Successful images:", metrics["total"])
    print("Failed images:", failed)
    print("Devices evaluated:", len(metrics["per_device"]))

    print(
        f"Top-1 accuracy: "
        f"{metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Top-3 accuracy: "
        f"{metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Top-1 accuracy: "
        f"{metrics['macro_top1_accuracy'] * 100:.2f}%"
    )

    print("\n--- Per-device Top-1 accuracy ---")

    for item in metrics["per_device"]:
        print(
            f"{item['device']}: "
            f"{item['top1_correct']}/{item['test_images']} "
            f"({item['top1_accuracy'] * 100:.2f}%)"
        )

    print("\n--- Saved files ---")
    print(predictions_path)
    print(per_device_path)
    print(confusion_path)
    print(summary_path)

    print("\n=============================================")


if __name__ == "__main__":
    main()
