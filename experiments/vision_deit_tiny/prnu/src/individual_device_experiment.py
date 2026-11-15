from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from scipy.fft import fft2, ifft2
from noise_residual import extract_noise_residual

# File location: experiments/vision_deit_tiny/prnu/src/
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = PROJECT_ROOT / "individual_device_demo" / "dataset"
REFERENCE_ROOT = DATASET_ROOT / "reference"
QUERY_ROOT = DATASET_ROOT / "query"
OUTPUT_ROOT = PROJECT_ROOT / "individual_device_demo" / "results"
FINGERPRINT_ROOT = PROJECT_ROOT / "individual_device_demo" / "fingerprints"

DEVICE_IDS = [f"D{i:02d}" for i in range(1, 23)]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CROP_SIZE = 512
PCE_EXCLUSION_RADIUS = 5


def center_crop(array):
    h, w = array.shape[:2]
    if h < CROP_SIZE or w < CROP_SIZE:
        raise ValueError(f"Image {w}x{h} is smaller than {CROP_SIZE}x{CROP_SIZE}")
    top, left = (h - CROP_SIZE) // 2, (w - CROP_SIZE) // 2
    return array[top:top + CROP_SIZE, left:left + CROP_SIZE]


def gray(array):
    return (0.299 * array[:, :, 0] + 0.587 * array[:, :, 1]
            + 0.114 * array[:, :, 2]).astype(np.float64)


def ncc(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a - np.mean(a), b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denom) if denom >= 1e-12 else 0.0


def pce(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a - np.mean(a), b - np.mean(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    corr = np.real(ifft2(fft2(a) * np.conj(fft2(b))))
    peak_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    peak_energy = float(corr[peak_idx]) ** 2
    h, w = corr.shape
    yy, xx = np.arange(h), np.arange(w)
    dy = np.minimum((yy - peak_idx[0]) % h, (peak_idx[0] - yy) % h)
    dx = np.minimum((xx - peak_idx[1]) % w, (peak_idx[1] - xx) % w)
    mask = ((dy[:, None] <= PCE_EXCLUSION_RADIUS)
            & (dx[None, :] <= PCE_EXCLUSION_RADIUS))
    sidelobes = corr.copy()
    sidelobes[mask] = 0
    count = sidelobes.size - np.count_nonzero(mask)
    return float(peak_energy / (np.sum(sidelobes ** 2) / count + 1e-12))


def images_in(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def build_fingerprint(device):
    paths = images_in(REFERENCE_ROOT / device)
    if not paths:
        raise RuntimeError(f"No reference images found for {device}")

    # Pixel-wise averaging requires matching original dimensions.
    groups = defaultdict(list)
    for path in paths:
        try:
            original, residual = extract_noise_residual(path)
            groups[original.shape[:2]].append((path, original, residual))
        except Exception as exc:
            print(f"  Skipped {path.name}: {exc}")

    if not groups:
        raise RuntimeError(f"No usable reference images for {device}")

    selected_shape = max(groups, key=lambda shape: len(groups[shape]))
    selected = groups[selected_shape]
    numerator = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    used = 0

    for path, original, residual in selected:
        try:
            original_crop = center_crop(original)
            residual_crop = center_crop(residual)
        except ValueError as exc:
            print(f"  Skipped {path.name}: {exc}")
            continue
        numerator += residual_crop.astype(np.float64) * original_crop
        denominator += original_crop.astype(np.float64) ** 2
        used += 1

    if not used:
        raise RuntimeError(f"No compatible reference images for {device}")

    fingerprint = numerator / (denominator + 1e-8)
    fingerprint -= np.mean(fingerprint, axis=(0, 1), keepdims=True)
    return fingerprint.astype(np.float32), used, len(paths), selected_shape


def generate_fingerprints():
    FINGERPRINT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    print("\n========== GENERATING 22 DEVICE FINGERPRINTS ==========")

    for device in DEVICE_IDS:
        fingerprint, used, available, shape = build_fingerprint(device)
        np.save(FINGERPRINT_ROOT / f"{device}.npy", fingerprint)
        summary[device] = {
            "reference_images_available": available,
            "reference_images_used": used,
            "selected_original_dimensions": list(shape),
            "fingerprint_shape": list(fingerprint.shape),
        }
        print(f"{device}: used {used}/{available} reference images")

    with (OUTPUT_ROOT / "fingerprint_generation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_fingerprints():
    result = {}
    for path in sorted(FINGERPRINT_ROOT.glob("*.npy")):
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.shape == (CROP_SIZE, CROP_SIZE, 3):
            result[path.stem] = arr
    if len(result) != len(DEVICE_IDS):
        raise RuntimeError(
            f"Expected 22 fingerprints, found {len(result)}. "
            "Check fingerprint generation output."
        )
    return result


def identify(image_path, fingerprints):
    original, residual = extract_noise_residual(image_path)
    original_crop, residual_crop = center_crop(original), center_crop(residual)
    residual_gray = gray(residual_crop)
    ranked = []

    for device, fingerprint in fingerprints.items():
        expected = gray(fingerprint.astype(np.float64) * original_crop.astype(np.float64))
        ranked.append({
            "device": device,
            "pce": pce(residual_gray, expected),
            "ncc": ncc(residual_gray, expected),
        })

    return sorted(ranked, key=lambda item: item["pce"], reverse=True)


def evaluate():
    fingerprints = load_fingerprints()
    rows, errors = [], []
    top1 = top3 = 0

    print("\n========== TESTING INDEPENDENT QUERY IMAGES ==========")
    for true_device in DEVICE_IDS:
        for image_path in images_in(QUERY_ROOT / true_device):
            try:
                ranked = identify(image_path, fingerprints)
                predicted = ranked[0]["device"]
                top3_devices = [x["device"] for x in ranked[:3]]
                top1 += int(predicted == true_device)
                top3 += int(true_device in top3_devices)
                rows.append({
                    "image": str(image_path.relative_to(DATASET_ROOT)),
                    "true_device": true_device,
                    "predicted_device": predicted,
                    "top1_correct": predicted == true_device,
                    "top3_correct": true_device in top3_devices,
                    "top1_pce": ranked[0]["pce"],
                    "top1_ncc": ranked[0]["ncc"],
                    "top2_device": ranked[1]["device"],
                    "top3_device": ranked[2]["device"],
                })
            except Exception as exc:
                errors.append({"image": str(image_path), "error": str(exc)})
        print(f"Completed {true_device}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUTPUT_ROOT / "query_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    confusion = {d: {p: 0 for p in DEVICE_IDS} for d in DEVICE_IDS}
    for row in rows:
        confusion[row["true_device"]][row["predicted_device"]] += 1

    with (OUTPUT_ROOT / "confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2)
    with (OUTPUT_ROOT / "evaluation_errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)

    count = len(rows)
    metrics = {
        "query_images_evaluated": count,
        "query_images_with_errors": len(errors),
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy_percent": round(100 * top1 / count, 2) if count else 0,
        "top3_accuracy_percent": round(100 * top3 / count, 2) if count else 0,
        "warning": (
            "Experimental closed-set PRNU matching. PCE scores are not "
            "calibrated probabilities and do not establish forensic-grade "
            "individual-device identification."
        ),
    }
    with (OUTPUT_ROOT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n========== RESULTS ==========")
    print(json.dumps(metrics, indent=2))
    print(f"Output folder: {OUTPUT_ROOT}")


def main():
    if not REFERENCE_ROOT.exists() or not QUERY_ROOT.exists():
        raise FileNotFoundError(f"Subset not found: {DATASET_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generate_fingerprints()
    evaluate()


if __name__ == "__main__":
    main()
