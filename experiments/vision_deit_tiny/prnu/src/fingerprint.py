
from pathlib import Path
from collections import defaultdict
import json

import numpy as np

from noise_residual import extract_noise_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "VISION"
FINGERPRINT_ROOT = PROJECT_ROOT / "prnu" / "fingerprints"
OUTPUT_ROOT = PROJECT_ROOT / "prnu" / "outputs"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}

# Limit initial experiment to a manageable number of images.
# Set to None later to use every available flat image.
MAX_IMAGES_PER_DEVICE = None


def collect_device_images():
    device_images = {}

    for device_folder in sorted(DATASET_ROOT.iterdir()):
        if not device_folder.is_dir():
            continue

        flat_folder = device_folder / "images" / "flat"

        if not flat_folder.exists():
            continue

        images = sorted(
            path for path in flat_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if MAX_IMAGES_PER_DEVICE is not None:
            images = images[:MAX_IMAGES_PER_DEVICE]

        if images:
            device_images[device_folder.name] = images

    return device_images


def estimate_fingerprint(image_paths):
    """
    Estimate a device fingerprint from flat images.

    Images with different dimensions are handled by grouping
    them by their original dimensions. The largest group is
    selected so that the fingerprint has a consistent shape.

    This avoids resizing, which can damage PRNU.
    """
    dimension_groups = defaultdict(list)

    for image_path in image_paths:
        try:
            _, residual = extract_noise_residual(image_path)

            height, width, _ = residual.shape

            dimension_groups[(height, width)].append(
                (image_path, residual)
            )

        except Exception as error:
            print(f"Skipping {image_path}: {error}")

    if not dimension_groups:
        return None, []

    selected_shape = max(
        dimension_groups,
        key=lambda shape: len(dimension_groups[shape])
    )

    selected_images = dimension_groups[selected_shape]

    numerator = np.zeros(
        (selected_shape[0], selected_shape[1], 3),
        dtype=np.float64
    )

    denominator = np.zeros_like(numerator)

    for image_path, residual in selected_images:
        original, _ = extract_noise_residual(image_path)

        # Intensity-weighted fingerprint accumulation.
        numerator += residual.astype(np.float64) * original
        denominator += original.astype(np.float64) ** 2

    fingerprint = numerator / (denominator + 1e-8)

    # Remove the global mean from each color channel.
    fingerprint -= np.mean(
        fingerprint,
        axis=(0, 1),
        keepdims=True
    )

    return fingerprint.astype(np.float32), [
        str(path) for path, _ in selected_images
    ]


def main():
    FINGERPRINT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    devices = collect_device_images()

    if not devices:
        raise RuntimeError("No device flat images were found.")

    summary = {}

    print("\n========== PRNU FINGERPRINT GENERATION ==========")

    for index, (device_name, image_paths) in enumerate(
        devices.items(), start=1
    ):
        print(
            f"\n[{index}/{len(devices)}] Processing "
            f"{device_name} ({len(image_paths)} images)"
        )

        fingerprint, used_images = estimate_fingerprint(
            image_paths
        )

        if fingerprint is None:
            print("No usable images. Skipping device.")
            continue

        output_path = (
            FINGERPRINT_ROOT / f"{device_name}.npy"
        )

        np.save(output_path, fingerprint)

        summary[device_name] = {
            "fingerprint_path": str(output_path),
            "fingerprint_shape": list(fingerprint.shape),
            "images_used": len(used_images),
            "source_category": "flat",
            "source_images": used_images
        }

        print("Fingerprint shape:", fingerprint.shape)
        print("Images used:", len(used_images))
        print("Saved:", output_path)

    summary_path = OUTPUT_ROOT / "fingerprint_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== GENERATION COMPLETE ==========")
    print("Devices processed:", len(summary))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
