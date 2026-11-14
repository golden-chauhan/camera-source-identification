
from pathlib import Path
from collections import Counter
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import VisionDataset, read_metadata
from model import create_model, MODEL_NAME


PROJECT_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "vision_metadata.csv"
)

MAPPING_PATH = (
    PROJECT_DIR / "data" / "processed" / "device_label_mapping.csv"
)

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_deit_tiny_balanced_finetune.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last_deit_tiny_balanced_finetune.pth"
)

BATCH_SIZE = 8
EPOCHS = 12
PATIENCE = 4

BACKBONE_LR = 1e-5
HEAD_LR = 3e-4

WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05

NUM_WORKERS = 0
RANDOM_SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history):
    path = OUTPUT_DIR / "training_history_balanced_finetune.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DEIT-TINY - BALANCED FINE-TUNING")
    print("=" * 60)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    metadata = read_metadata(METADATA_PATH)
    mapping = read_metadata(MAPPING_PATH)

    mapping.sort(key=lambda row: int(row["device_label"]))

    class_names = [
        row["device_id"] for row in mapping
    ]

    num_classes = len(class_names)

    split_counts = Counter(
        row["split"] for row in metadata
    )

    print("Architecture:", MODEL_NAME)
    print("Classes:", num_classes)
    print("Training:", split_counts["train"])
    print("Validation:", split_counts["validation"])
    print("Reserved test:", split_counts["test"])

    if split_counts["test"] != 500:
        raise ValueError("Test split must contain exactly 500 images.")

    train_dataset = VisionDataset(
        metadata,
        PROJECT_DIR,
        "train",
        training=True
    )

    validation_dataset = VisionDataset(
        metadata,
        PROJECT_DIR,
        "validation",
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print("\nLoading pretrained DeiT-Tiny...")

    model = create_model(
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    # IMPORTANT: All model parameters are trainable from epoch 1.
    for parameter in model.parameters():
        parameter.requires_grad = True

    head_parameters = list(
        model.get_classifier().parameters()
    )

    head_ids = {
        id(parameter) for parameter in head_parameters
    }

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_ids
    ]

    # Moderate class weighting.
    counts = Counter(
        int(row["device_label"])
        for row in train_dataset.samples
    )

    raw_weights = [
        np.sqrt(
            max(counts.values()) / counts[index]
        )
        for index in range(num_classes)
    ]

    mean_weight = sum(raw_weights) / len(raw_weights)

    class_weights = torch.tensor(
        [weight / mean_weight for weight in raw_weights],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR
            },
            {
                "params": head_parameters,
                "lr": HEAD_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_f1 = -1.0
    best_accuracy = 0.0
    no_improvement = 0

    history = []

    print("\nTraining configuration:")
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Backbone LR:", BACKBONE_LR)
    print("Head LR:", HEAD_LR)
    print("Weight decay:", WEIGHT_DECAY)
    print("Label smoothing:", LABEL_SMOOTHING)
    print("Dropout: 0.10")
    print("DropPath: 0.05")
    print("All Transformer blocks trainable: YES")
    print("Mixed precision:", use_amp)

    for epoch in range(EPOCHS):
        start_time = time.time()

        # ---------------- TRAINING ----------------
        model.train()

        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            train_predictions.extend(
                outputs.argmax(dim=1).detach().cpu().tolist()
            )

            train_targets.extend(
                labels.detach().cpu().tolist()
            )

        train_loss /= len(train_dataset)

        train_accuracy = accuracy_score(
            train_targets,
            train_predictions
        )

        # ---------------- VALIDATION ----------------
        model.eval()

        validation_loss = 0.0
        validation_predictions = []
        validation_targets = []

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                validation_loss += (
                    loss.item() * images.size(0)
                )

                validation_predictions.extend(
                    outputs.argmax(dim=1).cpu().tolist()
                )

                validation_targets.extend(
                    labels.cpu().tolist()
                )

        validation_loss /= len(validation_dataset)

        validation_accuracy = accuracy_score(
            validation_targets,
            validation_predictions
        )

        validation_f1 = f1_score(
            validation_targets,
            validation_predictions,
            average="macro",
            zero_division=0
        )

        scheduler.step(validation_f1)

        elapsed = time.time() - start_time

        print("\n" + "-" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")
        print(f"Validation macro F1: {validation_f1:.4f}")
        print(
            "Backbone LR:",
            optimizer.param_groups[0]["lr"]
        )
        print(
            "Head LR:",
            optimizer.param_groups[1]["lr"]
        )
        print(f"Time: {elapsed:.1f} seconds")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1
        })

        # Save best model by validation macro F1.
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_accuracy = validation_accuracy
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_validation_f1": best_f1,
                "best_validation_accuracy": best_accuracy,
                "image_size": 224,
                "drop_rate": 0.10,
                "drop_path_rate": 0.05
            }, BEST_CHECKPOINT)

            print("Best checkpoint saved.")

        else:
            no_improvement += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_validation_f1": best_f1,
            "class_names": class_names
        }, LAST_CHECKPOINT)

        save_history(history)

        if no_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    summary = {
        "architecture": MODEL_NAME,
        "classes": num_classes,
        "training_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "test_images_reserved": split_counts["test"],
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "checkpoint": str(BEST_CHECKPOINT)
    }

    with (
        OUTPUT_DIR / "training_summary_balanced_finetune.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== TRAINING COMPLETED ==========")
    print("Best validation accuracy:", round(best_accuracy, 4))
    print("Best validation macro F1:", round(best_f1, 4))
    print("Checkpoint:", BEST_CHECKPOINT)


if __name__ == "__main__":
    main()
