
from pathlib import Path
import csv

import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from dataset import VisionDataset, read_metadata
from model import create_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "vision_metadata.csv"
)

MAPPING_PATH = (
    PROJECT_DIR / "data" / "processed" / "device_label_mapping.csv"
)

# IMPORTANT: Correct balanced fine-tuning checkpoint
CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" /
    "best_deit_tiny_balanced_finetune.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== DEIT-TINY FINAL EVALUATION ==========")
    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    metadata = read_metadata(METADATA_PATH)
    mapping = read_metadata(MAPPING_PATH)

    mapping.sort(
        key=lambda row: int(row["device_label"])
    )

    class_names = [
        row["device_id"] for row in mapping
    ]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    checkpoint_class_names = checkpoint["class_names"]

    if checkpoint_class_names != class_names:
        raise ValueError(
            "Checkpoint class mapping does not match dataset mapping."
        )

    test_dataset = VisionDataset(
        metadata,
        PROJECT_DIR,
        "test",
        training=False
    )

    if len(test_dataset) != 500:
        raise ValueError(
            f"Expected exactly 500 test images, found {len(test_dataset)}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = create_model(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []
    all_top3 = []
    all_probabilities = []

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)

            probabilities = torch.softmax(
                outputs.float(),
                dim=1
            )

            predictions = probabilities.argmax(dim=1)

            top3 = probabilities.topk(
                k=min(3, len(class_names)),
                dim=1
            ).indices

            all_targets.extend(labels.tolist())

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_top3.extend(
                top3.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    # ---------------- METRICS ----------------

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    top3_accuracy = np.mean([
        target in predictions
        for target, predictions in zip(
            all_targets,
            all_top3
        )
    ])

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            labels=list(range(len(class_names))),
            zero_division=0
        )
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print("Checkpoint:", CHECKPOINT_PATH.name)
    print("Test images:", len(test_dataset))
    print("Test accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Top-3 accuracy:", round(float(top3_accuracy), 4))

    print("\nClassification report:")

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    # ---------------- PER-CLASS METRICS ----------------

    per_class_rows = []

    for index, name in enumerate(class_names):

        per_class_rows.append({
            "device": name,
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1_score": round(float(f1[index]), 4),
            "support": int(support[index])
        })

    write_rows(
        OUTPUT_DIR / "test_per_class_metrics.csv",
        per_class_rows,
        [
            "device",
            "precision",
            "recall",
            "f1_score",
            "support"
        ]
    )

    # ---------------- CONFUSION MATRIX ----------------

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    with (
        OUTPUT_DIR / "test_confusion_matrix.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["actual/predicted"] + class_names
        )

        for index, row in enumerate(matrix):

            writer.writerow(
                [class_names[index]] + row.tolist()
            )

    # ---------------- IMAGE PREDICTIONS ----------------

    prediction_rows = []

    test_rows = test_dataset.samples

    for row, target, prediction, probabilities in zip(
        test_rows,
        all_targets,
        all_predictions,
        all_probabilities
    ):

        prediction_rows.append({
            "filepath": row["filepath"],
            "actual_device": class_names[target],
            "predicted_device": class_names[prediction],
            "correct": int(target == prediction),
            "confidence": round(
                float(max(probabilities)),
                5
            )
        })

    write_rows(
        OUTPUT_DIR / "test_predictions.csv",
        prediction_rows,
        [
            "filepath",
            "actual_device",
            "predicted_device",
            "correct",
            "confidence"
        ]
    )

    # ---------------- FINAL SUMMARY ----------------

    summary = {
        "checkpoint": CHECKPOINT_PATH.name,
        "architecture": checkpoint.get(
            "model_name",
            "deit_tiny_patch16_224"
        ),
        "number_of_classes": len(class_names),
        "test_images": len(test_dataset),
        "test_accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "top3_accuracy": round(float(top3_accuracy), 4)
    }

    import json

    with (
        OUTPUT_DIR / "final_test_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nEvaluation completed successfully.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
