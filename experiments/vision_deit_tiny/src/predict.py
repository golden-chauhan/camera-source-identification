
from pathlib import Path
import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import create_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "best_deit_tiny_regularized.pth"
)


def main():
    parser = argparse.ArgumentParser(
        description="VISION DeiT-Tiny camera device prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image to classify"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_transforms(training=False)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(tensor)

        probabilities = torch.softmax(outputs.float(), dim=1)[0]

    top_values, top_indices = probabilities.topk(
        min(3, len(class_names))
    )

    print("\n========== PREDICTION ==========")
    print("Input:", image_path)
    print("Predicted device:", class_names[top_indices[0].item()])
    print("Confidence:", f"{top_values[0].item() * 100:.2f}%")

    print("\nTop predictions:")

    for value, index in zip(top_values, top_indices):
        print(
            f"{class_names[index.item()]}: "
            f"{value.item() * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
