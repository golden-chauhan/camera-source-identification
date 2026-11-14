
from pathlib import Path
from collections import defaultdict
import hashlib
import re

ROOT = Path("data/raw/VISION")

def collect_images(category):
    result = defaultdict(dict)

    for device in ROOT.iterdir():
        folder = device / "images" / category

        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            # Remove category token from filename
            normalized = re.sub(
                rf"_{re.escape(category)}_",
                "_",
                path.stem,
                flags=re.IGNORECASE
            )

            result[device.name][normalized] = path

    return result


flat = collect_images("flat")
nat = collect_images("nat")

total_name_matches = 0
total_hash_matches = 0

print("\n========== NORMALIZED FLAT / NAT COMPARISON ==========\n")

for device in sorted(set(flat) | set(nat)):
    flat_files = flat[device]
    nat_files = nat[device]

    common = set(flat_files) & set(nat_files)

    hash_matches = 0

    for name in common:
        flat_hash = hashlib.sha256(flat_files[name].read_bytes()).hexdigest()
        nat_hash = hashlib.sha256(nat_files[name].read_bytes()).hexdigest()

        if flat_hash == nat_hash:
            hash_matches += 1

    total_name_matches += len(common)
    total_hash_matches += hash_matches

    print(
        f"{device}: "
        f"normalized_matches={len(common)}, "
        f"identical_files={hash_matches}"
    )

print("\nTotal normalized filename matches:", total_name_matches)
print("Total byte-identical image pairs:", total_hash_matches)
print("\n========== COMPARISON COMPLETE ==========")
