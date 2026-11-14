
from pathlib import Path

import numpy as np
import pywt
from PIL import Image


def load_image(image_path):
    """
    Load an image as an RGB float32 array.

    Original pixel dimensions are preserved.
    No resizing or image compression is performed.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_array = np.asarray(image, dtype=np.float32)

    return image_array


def wavelet_denoise_channel(channel, wavelet="db8", level=3):
    """
    Denoise a single image channel using wavelet thresholding.
    """
    height, width = channel.shape

    wavelet_object = pywt.Wavelet(wavelet)

    max_level = pywt.dwtn_max_level(
        channel.shape,
        wavelet_object
    )

    actual_level = min(level, max_level)

    if actual_level < 1:
        raise ValueError(
            f"Image dimensions {channel.shape} are too small "
            "for wavelet decomposition."
        )

    coefficients = pywt.wavedec2(
        channel,
        wavelet_object,
        level=actual_level,
        mode="symmetric"
    )

    detail_coefficients = coefficients[1:]

    finest_details = detail_coefficients[-1]
    diagonal_details = finest_details[2]

    # Robust noise estimation using Median Absolute Deviation
    sigma = (
        np.median(np.abs(diagonal_details))
        / 0.6745
    )

    threshold = sigma * np.sqrt(
        2.0 * np.log(channel.size)
    )

    denoised_coefficients = [coefficients[0]]

    for details in detail_coefficients:
        thresholded_details = tuple(
            pywt.threshold(
                detail,
                threshold,
                mode="soft"
            )
            for detail in details
        )

        denoised_coefficients.append(
            thresholded_details
        )

    denoised = pywt.waverec2(
        denoised_coefficients,
        wavelet_object,
        mode="symmetric"
    )

    # Reconstruction can differ by a few pixels.
    denoised = denoised[:height, :width]

    return denoised.astype(np.float32)


def extract_noise_residual(
    image_path,
    wavelet="db8",
    level=3
):
    """
    Extract RGB noise residual from an image.

    Returns:
        original: Original RGB image as float32.
        residual: Image minus wavelet-denoised image.
    """
    original = load_image(image_path)

    residual = np.zeros_like(
        original,
        dtype=np.float32
    )

    for channel_index in range(3):
        channel = original[:, :, channel_index]

        denoised = wavelet_denoise_channel(
            channel,
            wavelet=wavelet,
            level=level
        )

        residual[:, :, channel_index] = (
            channel - denoised
        )

    return original, residual


def main():
    project_root = Path(__file__).resolve().parents[2]

    image_root = (
        project_root
        / "data"
        / "raw"
        / "VISION"
    )

    # Select one flat image for the initial test.
    candidates = sorted(image_root.glob(
        "D01_*/images/flat/*.jpg"
    ))

    if not candidates:
        raise FileNotFoundError(
            "No flat JPEG image found for testing."
        )

    image_path = candidates[0]

    print("\n========== PRNU RESIDUAL TEST ==========")
    print("Image:", image_path)

    original, residual = extract_noise_residual(
        image_path
    )

    print("Original shape:", original.shape)
    print("Residual shape:", residual.shape)
    print("Original dtype:", original.dtype)
    print("Residual dtype:", residual.dtype)
    print(
        "Residual mean:",
        float(np.mean(residual))
    )
    print(
        "Residual standard deviation:",
        float(np.std(residual))
    )
    print(
        "Residual minimum:",
        float(np.min(residual))
    )
    print(
        "Residual maximum:",
        float(np.max(residual))
    )

    print("\nResidual extraction successful.")
    print("========================================")


if __name__ == "__main__":
    main()
