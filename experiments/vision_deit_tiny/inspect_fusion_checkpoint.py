
from pathlib import Path
import torch

checkpoint_path = (
    Path("checkpoints")
    / "best_deit_tiny_balanced_finetune.pth"
)

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path.resolve())

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("\n========== DEIT CHECKPOINT INSPECTION ==========")
print("Checkpoint:", checkpoint_path.resolve())
print("Object type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nTop-level keys:")

    for key, value in checkpoint.items():
        print(f"\n{key}:")
        print("  Type:", type(value))

        if isinstance(value, dict):
            print("  Number of entries:", len(value))
            print("  First keys:", list(value.keys())[:8])

            if any(
                word in str(key).lower()
                for word in ["class", "label", "mapping"]
            ):
                print("  Mapping:", value)

        elif isinstance(value, (list, tuple)):
            print("  Length:", len(value))
            print("  First entries:", value[:10])

        elif isinstance(value, torch.Tensor):
            print("  Shape:", tuple(value.shape))

        else:
            print("  Value:", str(value)[:300])

print("\n========== INSPECTION COMPLETE ==========")
