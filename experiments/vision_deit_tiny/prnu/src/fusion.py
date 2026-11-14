
from pathlib import Path
import sys
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "prnu" / "src"))

from dataset import VisionDataset, read_metadata
from model import create_model
from matching import load_fingerprints, prepare_query

from scipy.fft import fft2, ifft2


CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "best_deit_tiny_balanced_finetune.pth"
)

METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "vision_metadata.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "device_label_mapping.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "prnu" / "outputs" / "fusion"

BATCH_SIZE = 8
CACHE_INTERVAL = 10

CROP_SIZE = 512
PCE_EXCLUSION_RADIUS = 5

# Explicitly confirmed from your metadata.
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

FUSION_WEIGHTS = np.arange(0.0, 1.01, 0.05)


# ============================================================
# CSV UTILITIES
# ============================================================

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MODEL LOADING
# ============================================================

def load_deit(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = create_model(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, class_names


def generate_deit_scores(dataset, model, device, split_name):
    """Generate DeiT probabilities using the original preprocessing."""

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    all_scores = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                logits = model(images)

            probabilities = torch.softmax(
                logits.float(),
                dim=1
            )

            all_scores.extend(
                probabilities.cpu().numpy()
            )

            print(
                f"{split_name} DeiT: "
                f"{batch_index + 1}/{len(loader)} batches",
                end="\r"
            )

    print()

    return np.asarray(all_scores, dtype=np.float64)


# ============================================================
# PRNU UTILITIES
# ============================================================

def center_crop(array):
    height, width = array.shape[:2]

    if height < CROP_SIZE or width < CROP_SIZE:
        raise ValueError(
            f"Image too small for {CROP_SIZE}x{CROP_SIZE} crop."
        )

    top = (height - CROP_SIZE) // 2
    left = (width - CROP_SIZE) // 2

    return array[
        top:top + CROP_SIZE,
        left:left + CROP_SIZE
    ]


def to_grayscale(array):
    return (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    ).astype(np.float64)


def calculate_pce_from_fft(query_fft, expected_pattern):
    """Calculate PCE while reusing the query image FFT."""

    expected = expected_pattern.astype(np.float64)
    expected -= np.mean(expected)

    if np.std(expected) < 1e-12:
        return 0.0

    correlation = np.real(
        ifft2(
            query_fft * np.conj(fft2(expected))
        )
    )

    peak_index = np.unravel_index(
        np.argmax(np.abs(correlation)),
        correlation.shape
    )

    peak_value = float(correlation[peak_index])
    peak_energy = peak_value ** 2

    height, width = correlation.shape
    peak_y, peak_x = peak_index

    y_indices = np.arange(height)
    x_indices = np.arange(width)

    y_distance = np.minimum(
        (y_indices - peak_y) % height,
        (peak_y - y_indices) % height
    )

    x_distance = np.minimum(
        (x_indices - peak_x) % width,
        (peak_x - x_indices) % width
    )

    exclusion = (
        (y_distance[:, None] <= PCE_EXCLUSION_RADIUS)
        & (x_distance[None, :] <= PCE_EXCLUSION_RADIUS)
    )

    sidelobes = correlation.copy()
    sidelobes[exclusion] = 0.0

    valid_count = sidelobes.size - np.count_nonzero(exclusion)

    if valid_count <= 0:
        return 0.0

    sidelobe_energy = (
        np.sum(sidelobes ** 2) / valid_count
    )

    return float(
        peak_energy / (sidelobe_energy + 1e-12)
    )


def calculate_image_prnu_scores(
    image_path,
    fingerprints,
    class_to_index,
    number_of_classes
):
    """
    Calculate PCE scores against every device fingerprint.

    The image residual and its FFT are computed once.
    """
    original, residual = prepare_query(image_path)

    original_crop = center_crop(original)
    residual_crop = center_crop(residual)

    residual_gray = to_grayscale(residual_crop)
    residual_gray -= np.mean(residual_gray)

    if np.std(residual_gray) < 1e-12:
        raise ValueError("Image residual has insufficient variation.")

    query_fft = fft2(residual_gray)

    scores = np.zeros(
        number_of_classes,
        dtype=np.float64
    )

    compatible = 0

    for device_name, fingerprint in fingerprints.items():

        if device_name not in class_to_index:
            continue

        if (
            fingerprint.shape[0] < CROP_SIZE
            or fingerprint.shape[1] < CROP_SIZE
        ):
            continue

        fingerprint_crop = center_crop(fingerprint)

        expected_rgb = (
            fingerprint_crop.astype(np.float64)
            * original_crop.astype(np.float64)
        )

        expected_gray = to_grayscale(expected_rgb)

        pce = calculate_pce_from_fft(
            query_fft,
            expected_gray
        )

        scores[class_to_index[device_name]] = max(0.0, pce)
        compatible += 1

    if compatible == 0:
        raise RuntimeError(
            "No compatible PRNU fingerprints for this image."
        )

    return scores


# ============================================================
# RESUMABLE PRNU CACHE
# ============================================================

def generate_prnu_scores(
    dataset,
    class_names,
    fingerprints,
    split_name
):
    """
    Save partial results and resume unfinished images safely.

    A row is marked complete only after all device scores
    have been calculated successfully.
    """
    cache_path = (
        OUTPUT_DIR / f"{split_name}_prnu_cache.npz"
    )

    current_paths = [
        row["filepath"] for row in dataset.samples
    ]

    number_of_images = len(current_paths)
    number_of_classes = len(class_names)

    scores = np.full(
        (number_of_images, number_of_classes),
        np.nan,
        dtype=np.float64
    )

    completed = np.zeros(
        number_of_images,
        dtype=bool
    )

    if cache_path.exists():

        try:
            cache = np.load(
                cache_path,
                allow_pickle=False
            )

            old_paths = cache["paths"].tolist()
            old_classes = cache["classes"].tolist()
            old_scores = cache["scores"]
            old_completed = cache["completed"]

            if (
                old_paths == current_paths
                and old_classes == class_names
                and old_scores.shape == scores.shape
                and old_completed.shape == completed.shape
            ):
                scores = old_scores.astype(np.float64)
                completed = old_completed.astype(bool)

                print(
                    f"Resuming {split_name} PRNU cache: "
                    f"{completed.sum()}/{number_of_images} complete."
                )

            else:
                print(
                    "Existing cache does not match this dataset. "
                    "Starting a new cache."
                )

        except Exception as error:
            print("Could not load cache:", error)

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    def save_progress():
        np.savez_compressed(
            cache_path,
            scores=scores,
            completed=completed,
            paths=np.asarray(current_paths),
            classes=np.asarray(class_names)
        )

    remaining = int((~completed).sum())

    print(
        f"{split_name} PRNU images remaining:",
        remaining
    )

    for index, row in enumerate(dataset.samples):

        if completed[index]:
            continue

        image_path = PROJECT_ROOT / row["filepath"]

        print(
            f"\n{split_name} PRNU "
            f"{index + 1}/{number_of_images}: "
            f"{row['device_id']}"
        )

        try:
            image_scores = calculate_image_prnu_scores(
                image_path,
                fingerprints,
                class_to_index,
                number_of_classes
            )

            scores[index] = image_scores
            completed[index] = True

        except Exception as error:
            save_progress()

            raise RuntimeError(
                f"PRNU failed for {image_path}: {error}"
            ) from error

        if (
            (index + 1) % CACHE_INTERVAL == 0
            or index + 1 == number_of_images
        ):
            save_progress()

            print(
                f"Cache saved: "
                f"{completed.sum()}/{number_of_images} complete."
            )

    save_progress()

    if not np.all(completed):
        raise RuntimeError(
            "PRNU cache contains unfinished images."
        )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            "PRNU score matrix contains invalid values."
        )

    return scores


# ============================================================
# TARGETS AND LABEL ALIGNMENT
# ============================================================

def get_targets(dataset, class_names):
    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    targets = []

    for row in dataset.samples:

        device_name = row["device_id"]

        if device_name not in class_to_index:
            raise ValueError(
                f"Unknown device: {device_name}"
            )

        target = class_to_index[device_name]

        if int(row["device_label"]) != target:
            raise ValueError(
                f"Label mismatch for {device_name}: "
                f"metadata={row['device_label']}, "
                f"checkpoint={target}"
            )

        targets.append(target)

    return np.asarray(targets, dtype=np.int64)


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def standardize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)

    means = scores.mean(axis=1, keepdims=True)
    stds = scores.std(axis=1, keepdims=True)

    return (
        scores - means
    ) / np.maximum(stds, 1e-8)


def normalize_deit(probabilities):
    return standardize_scores(
        np.log(np.maximum(probabilities, 1e-12))
    )


def normalize_prnu(pce_scores):
    return standardize_scores(
        np.log1p(np.maximum(pce_scores, 0.0))
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(targets, scores):
    predictions = np.argmax(scores, axis=1)

    top3 = np.argsort(
        scores,
        axis=1
    )[:, -3:]

    top3_accuracy = np.mean([
        target in candidates
        for target, candidates in zip(targets, top3)
    ])

    return {
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "top3_accuracy": float(top3_accuracy),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "predictions": predictions
    }


# ============================================================
# FUSION WEIGHT SELECTION
# ============================================================

def select_fusion_weight(
    targets,
    deit_scores,
    prnu_scores
):
    results = []

    best_weight = 0.0
    best_macro_f1 = -1.0
    best_accuracy = -1.0

    for weight in FUSION_WEIGHTS:

        combined = (
            (1.0 - weight) * deit_scores
            + weight * prnu_scores
        )

        metrics = calculate_metrics(
            targets,
            combined
        )

        results.append({
            "prnu_weight": float(weight),
            "deit_weight": float(1.0 - weight),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top3_accuracy": metrics["top3_accuracy"]
        })

        if (
            metrics["macro_f1"] > best_macro_f1
            or (
                np.isclose(
                    metrics["macro_f1"],
                    best_macro_f1
                )
                and metrics["accuracy"] > best_accuracy
            )
        ):
            best_macro_f1 = metrics["macro_f1"]
            best_accuracy = metrics["accuracy"]
            best_weight = float(weight)

    save_csv(
        OUTPUT_DIR / "validation_weight_search.csv",
        results
    )

    return best_weight


# ============================================================
# RESULT EXPORT
# ============================================================

def save_evaluation(
    method_name,
    targets,
    scores,
    class_names
):
    metrics = calculate_metrics(targets, scores)

    predictions = metrics["predictions"]

    per_device = []

    for index, device_name in enumerate(class_names):

        mask = targets == index

        if not np.any(mask):
            continue

        total = int(mask.sum())
        correct = int(
            np.sum(predictions[mask] == targets[mask])
        )

        per_device.append({
            "device": device_name,
            "images": total,
            "correct": correct,
            "accuracy": correct / total
        })

    save_csv(
        OUTPUT_DIR / f"{method_name}_test_per_device.csv",
        per_device
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names)))
    )

    with open(
        OUTPUT_DIR / f"{method_name}_test_confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + class_names)

        for index, row in enumerate(matrix):
            writer.writerow(
                [class_names[index]] + row.tolist()
            )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== DEIT + PRNU FUSION ==========")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Compute device:", device)

    metadata = read_metadata(METADATA_PATH)

    available_splits = {
        row["split"].strip().lower()
        for row in metadata
    }

    print("Available splits:", sorted(available_splits))

    if VALIDATION_SPLIT not in available_splits:
        raise ValueError(
            f"Validation split '{VALIDATION_SPLIT}' not found."
        )

    if TEST_SPLIT not in available_splits:
        raise ValueError(
            f"Test split '{TEST_SPLIT}' not found."
        )

    model, class_names = load_deit(device)

    mapping = read_metadata(MAPPING_PATH)
    mapping.sort(key=lambda row: int(row["device_label"]))

    mapping_names = [
        row["device_id"] for row in mapping
    ]

    if mapping_names != class_names:
        raise ValueError(
            "Checkpoint and mapping class order do not match."
        )

    fingerprints = load_fingerprints()

    missing = set(class_names) - set(fingerprints.keys())

    if missing:
        raise ValueError(
            "Missing fingerprints: " + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n========== VALIDATION ==========")

    validation_dataset = VisionDataset(
        metadata,
        PROJECT_ROOT,
        VALIDATION_SPLIT,
        training=False
    )

    validation_targets = get_targets(
        validation_dataset,
        class_names
    )

    print("Validation images:", len(validation_dataset))

    validation_deit_raw = generate_deit_scores(
        validation_dataset,
        model,
        device,
        "Validation"
    )

    validation_prnu_raw = generate_prnu_scores(
        validation_dataset,
        class_names,
        fingerprints,
        "validation"
    )

    validation_deit = normalize_deit(
        validation_deit_raw
    )

    validation_prnu = normalize_prnu(
        validation_prnu_raw
    )

    best_weight = select_fusion_weight(
        validation_targets,
        validation_deit,
        validation_prnu
    )

    print("\nSelected PRNU weight:", best_weight)
    print("Selected DeiT weight:", 1.0 - best_weight)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n========== TEST ==========")

    test_dataset = VisionDataset(
        metadata,
        PROJECT_ROOT,
        TEST_SPLIT,
        training=False
    )

    if len(test_dataset) != 500:
        raise ValueError(
            f"Expected 500 test images, found {len(test_dataset)}"
        )

    test_targets = get_targets(
        test_dataset,
        class_names
    )

    print("Test images:", len(test_dataset))

    test_deit_raw = generate_deit_scores(
        test_dataset,
        model,
        device,
        "Test"
    )

    test_prnu_raw = generate_prnu_scores(
        test_dataset,
        class_names,
        fingerprints,
        "test"
    )

    test_deit = normalize_deit(test_deit_raw)
    test_prnu = normalize_prnu(test_prnu_raw)

    test_fusion = (
        (1.0 - best_weight) * test_deit
        + best_weight * test_prnu
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    deit_metrics = save_evaluation(
        "deit",
        test_targets,
        test_deit,
        class_names
    )

    prnu_metrics = save_evaluation(
        "prnu",
        test_targets,
        test_prnu,
        class_names
    )

    fusion_metrics = save_evaluation(
        "fusion",
        test_targets,
        test_fusion,
        class_names
    )

    comparison = {
        "test_images": len(test_targets),
        "validation_split": VALIDATION_SPLIT,
        "selected_prnu_weight": best_weight,
        "selected_deit_weight": 1.0 - best_weight,
        "selection_metric": "validation_macro_f1",
        "deit": {
            key: value
            for key, value in deit_metrics.items()
            if key != "predictions"
        },
        "prnu": {
            key: value
            for key, value in prnu_metrics.items()
            if key != "predictions"
        },
        "fusion": {
            key: value
            for key, value in fusion_metrics.items()
            if key != "predictions"
        }
    }

    summary_path = (
        OUTPUT_DIR / "fusion_final_comparison.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(comparison, file, indent=4)

    print("\n========== FINAL COMPARISON ==========")

    for name in ("deit", "prnu", "fusion"):

        result = comparison[name]

        print(f"\n{name.upper()}")
        print(f"Top-1: {result['accuracy'] * 100:.2f}%")
        print(f"Top-3: {result['top3_accuracy'] * 100:.2f}%")
        print(f"Macro F1: {result['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1: {result['weighted_f1'] * 100:.2f}%")

    print("\nPRNU weight:", best_weight)
    print("DeiT weight:", 1.0 - best_weight)

    print("\nResults saved in:", OUTPUT_DIR)
    print("Summary:", summary_path)

    print("\n========== FUSION COMPLETE ==========")


if __name__ == "__main__":
    main()
