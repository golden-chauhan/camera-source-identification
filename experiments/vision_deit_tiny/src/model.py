
import timm

MODEL_NAME = "deit_tiny_patch16_224"
IMAGE_SIZE = 224


def create_model(num_classes, pretrained=True):
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.10,
        drop_path_rate=0.05
    )
