
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import VisionDataset, read_metadata
from model import create_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "vision_metadata.csv"
)

MAPPING_PATH = (
    PROJECT_DIR / "data" / "processed" / "device_label_mapping.csv"
)

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" /
    "best_deit_tiny_balanced_finetune.pth"
)

BATCH_SIZE = 8


def evaluate_split(model, loader, device):
    targets = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)

            predictions.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )
            targets.extend(labels.tolist())

    accuracy = accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "targets": targets,
        "predictions": predictions
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    metadata = read_metadata(METADATA_PATH)
    mapping = read_metadata(MAPPING_PATH)

    mapping.sort(
        key=lambda row: int(row["device_label"])
    )

    mapping_names = [
        row["device_id"] for row in mapping
    ]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    checkpoint_names = checkpoint["class_names"]

    print("\n========== CHECKPOINT VERIFICATION ==========")
    print("Model:", checkpoint["model_name"])
    print("Checkpoint classes:", len(checkpoint_names))
    print("Mapping classes:", len(mapping_names))
    print(
        "Class mappings identical:",
        checkpoint_names == mapping_names
    )

    if checkpoint_names != mapping_names:
        raise ValueError(
            "Checkpoint class order differs from current mapping."
        )

    model = create_model(
        num_classes=len(checkpoint_names),
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("\nCheckpoint loaded:", CHECKPOINT_PATH)

    for split in ("validation", "test"):
        dataset = VisionDataset(
            metadata,
            PROJECT_DIR,
            split,
            training=False
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

        result = evaluate_split(model, loader, device)

        print(f"\n========== {split.upper()} RESULTS ==========")
        print("Images:", len(dataset))
        print("Accuracy:", round(result["accuracy"], 4))
        print("Macro F1:", round(result["macro_f1"], 4))
        print("Weighted F1:", round(result["weighted_f1"], 4))

        correct = sum(
            actual == predicted
            for actual, predicted in zip(
                result["targets"],
                result["predictions"]
            )
        )

        print(f"Correct: {correct}/{len(dataset)}")

        print("\nPer-class accuracy:")

        for label, name in enumerate(checkpoint_names):
            class_indices = [
                index
                for index, actual in enumerate(result["targets"])
                if actual == label
            ]

            class_correct = sum(
                result["predictions"][index] == label
                for index in class_indices
            )

            total = len(class_indices)

            accuracy = (
                class_correct / total if total else 0
            )

            print(
                f"{name}: "
                f"{class_correct}/{total} "
                f"({accuracy * 100:.2f}%)"
            )


if __name__ == "__main__":
    main()
