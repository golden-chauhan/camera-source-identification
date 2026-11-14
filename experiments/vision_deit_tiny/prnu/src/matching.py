
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2

from noise_residual import extract_noise_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINGERPRINT_DIR = PROJECT_ROOT / "prnu" / "fingerprints_deploy"

CROP_SIZE = 512
PCE_EXCLUSION_RADIUS = 5


def load_fingerprints():
    """Load all saved device fingerprints."""
    fingerprints = {}

    for path in sorted(FINGERPRINT_DIR.glob("*.npy")):
        try:
            fingerprint = np.load(
                path,
                allow_pickle=False
            ).astype(np.float32)

            if fingerprint.ndim != 3 or fingerprint.shape[2] != 3:
                print(f"Skipping invalid fingerprint: {path.name}")
                continue

            fingerprints[path.stem] = fingerprint

        except Exception as error:
            print(f"Could not load {path.name}: {error}")

    if not fingerprints:
        raise RuntimeError(
            f"No valid fingerprints found in {FINGERPRINT_DIR}"
        )

    return fingerprints


def to_grayscale(array):
    """Convert an RGB array to grayscale."""
    return (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    ).astype(np.float64)


def center_crop(array, crop_size=CROP_SIZE):
    """
    Extract a center crop without resizing or interpolation.
    """
    height, width = array.shape[:2]

    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image dimensions {width}x{height} "
            f"are smaller than crop size {crop_size}."
        )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    return array[
        top:top + crop_size,
        left:left + crop_size
    ]


def calculate_ncc(residual, expected_pattern):
    """Calculate normalized correlation at zero displacement."""
    residual = residual.astype(np.float64)
    expected_pattern = expected_pattern.astype(np.float64)

    residual = residual - np.mean(residual)
    expected_pattern = expected_pattern - np.mean(expected_pattern)

    denominator = (
        np.linalg.norm(residual)
        * np.linalg.norm(expected_pattern)
    )

    if denominator < 1e-12:
        return 0.0

    return float(
        np.sum(residual * expected_pattern) / denominator
    )


def calculate_pce(residual, expected_pattern):
    """
    Calculate Peak-to-Correlation Energy.

    The peak is searched over circular spatial shifts.
    """
    residual = residual.astype(np.float64)
    expected_pattern = expected_pattern.astype(np.float64)

    residual -= np.mean(residual)
    expected_pattern -= np.mean(expected_pattern)

    if (
        np.std(residual) < 1e-12
        or np.std(expected_pattern) < 1e-12
    ):
        return 0.0, 0.0

    correlation_map = np.real(
        ifft2(
            fft2(residual)
            * np.conj(fft2(expected_pattern))
        )
    )

    peak_index = np.unravel_index(
        np.argmax(np.abs(correlation_map)),
        correlation_map.shape
    )

    peak_value = float(correlation_map[peak_index])
    peak_energy = peak_value ** 2

    height, width = correlation_map.shape
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

    exclusion_mask = (
        (y_distance[:, None] <= PCE_EXCLUSION_RADIUS)
        & (x_distance[None, :] <= PCE_EXCLUSION_RADIUS)
    )

    sidelobes = correlation_map.copy()
    sidelobes[exclusion_mask] = 0.0

    valid_count = sidelobes.size - np.count_nonzero(
        exclusion_mask
    )

    if valid_count <= 0:
        return 0.0, peak_value

    sidelobe_energy = (
        np.sum(sidelobes ** 2) / valid_count
    )

    pce = peak_energy / (sidelobe_energy + 1e-12)

    return float(pce), peak_value


def prepare_query(image_path):
    """
    Load a query image and extract its residual once.

    Returns original RGB image and RGB noise residual.
    """
    original, residual = extract_noise_residual(image_path)

    if original.shape != residual.shape:
        raise ValueError(
            "Original image and residual dimensions do not match."
        )

    if (
        original.shape[0] < CROP_SIZE
        or original.shape[1] < CROP_SIZE
    ):
        raise ValueError(
            f"Input image is too small for a {CROP_SIZE}x{CROP_SIZE} crop."
        )

    return original, residual


def compare_prepared_image(original, residual, fingerprint):
    """
    Compare a prepared image against one device fingerprint.

    A center crop is taken from both arrays. No resizing is used.
    """
    if (
        fingerprint.shape[0] < CROP_SIZE
        or fingerprint.shape[1] < CROP_SIZE
    ):
        return None

    original_crop = center_crop(original)
    residual_crop = center_crop(residual)
    fingerprint_crop = center_crop(fingerprint)

    # Expected PRNU pattern: K multiplied by image intensity.
    expected_rgb = (
        fingerprint_crop.astype(np.float64)
        * original_crop.astype(np.float64)
    )

    residual_gray = to_grayscale(residual_crop)
    expected_gray = to_grayscale(expected_rgb)

    ncc = calculate_ncc(
        residual_gray,
        expected_gray
    )

    pce, peak = calculate_pce(
        residual_gray,
        expected_gray
    )

    return {
        "ncc": ncc,
        "pce": pce,
        "peak_correlation": peak
    }


def identify_image(image_path, fingerprints=None):
    """
    Compare one input image against all compatible fingerprints.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if fingerprints is None:
        fingerprints = load_fingerprints()

    original, residual = prepare_query(image_path)

    results = []
    skipped_devices = []

    for device_name, fingerprint in fingerprints.items():
        try:
            result = compare_prepared_image(
                original,
                residual,
                fingerprint
            )

            if result is None:
                skipped_devices.append(device_name)
                continue

            results.append({
                "device": device_name,
                **result
            })

        except Exception as error:
            print(
                f"Comparison failed for {device_name}: {error}"
            )
            skipped_devices.append(device_name)

    results.sort(
        key=lambda item: item["pce"],
        reverse=True
    )

    return results, skipped_devices


def main():
    parser = argparse.ArgumentParser(
        description="PRNU-based camera device identification."
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the test image."
    )

    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        dataset_root = PROJECT_ROOT / "data" / "raw" / "VISION"

        candidates = sorted(
            dataset_root.glob(
                "D01_*/images/nat/*.jpg"
            )
        )

        if not candidates:
            raise FileNotFoundError(
                "No D01 natural images found for testing."
            )

        image_path = candidates[0]

    print("\n========== PRNU IMAGE MATCHING ==========")
    print("Input image:", image_path)
    print("Crop size:", f"{CROP_SIZE} x {CROP_SIZE}")

    fingerprints = load_fingerprints()

    print("Fingerprints loaded:", len(fingerprints))

    results, skipped = identify_image(
        image_path,
        fingerprints
    )

    if not results:
        print("\nNo compatible fingerprints found.")
        return

    print("\n--- Ranked device matches (PCE) ---")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:02d}. {result['device']}"
            f" | PCE: {result['pce']:.6f}"
            f" | NCC: {result['ncc']:.6f}"
        )

    print("\n--- Matching summary ---")
    print("Compatible devices:", len(results))
    print("Skipped devices:", len(skipped))

    if skipped:
        print("Skipped list:", ", ".join(skipped))

    print("\nHighest-scoring candidate:", results[0]["device"])

    print(
        "\nIMPORTANT: PCE ranking is not a calibrated "
        "identification probability."
    )

    print("=========================================")


if __name__ == "__main__":
    main()
