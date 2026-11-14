
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_metadata(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_transforms(training=False):
    if training:
        return transforms.Compose([
            transforms.Resize(240),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD
            )
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])


class VisionDataset(Dataset):
    def __init__(self, metadata, project_dir, split, training=False):
        self.project_dir = Path(project_dir)

        self.samples = [
            row for row in metadata
            if row["split"] == split
        ]

        self.transform = get_transforms(training)

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

        for row in self.samples:
            row["device_label"] = int(row["device_label"])

            image_path = self.project_dir / row["filepath"]

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image_path = self.project_dir / row["filepath"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, row["device_label"]
