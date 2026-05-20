"""
augment_to_target.py
====================
Bulk augment existing per-class .npy files until each class reaches a target count.

Usage:
    python augment_to_target.py --library_dir 16_04_Training_Data --target 250
"""

import argparse
from pathlib import Path
import numpy as np

from generate_training_data import augment_point_cloud


def augment_class(class_dir: Path, class_name: str, target: int) -> int:
    """Augment samples in class_dir until there are `target` total files."""
    existing = sorted(class_dir.glob(f"{class_name}_*.npy"))
    n_existing = len(existing)

    if n_existing == 0:
        print(f"  [{class_name}] no base samples found, skipping")
        return 0
    if n_existing >= target:
        print(f"  [{class_name}] already has {n_existing} samples (>= {target}), skipping")
        return 0

    # Load all base samples (originals, not augmented). Base samples are numbered
    # starting at 1 — we use *all* existing files as seeds for augmentation.
    base_pts = [np.load(f).astype(np.float32) for f in existing]

    next_id = n_existing + 1
    needed = target - n_existing
    generated = 0

    print(f"  [{class_name}] {n_existing} → {target} (generating {needed} augmentations)")

    while generated < needed:
        # Cycle through the base samples to get balanced augmentations
        base = base_pts[generated % len(base_pts)]
        aug = augment_point_cloud(base)
        out_path = class_dir / f"{class_name}_{next_id}.npy"
        np.save(out_path, aug)
        next_id += 1
        generated += 1

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Augment each class up to a target sample count."
    )
    parser.add_argument("--library_dir", type=str, required=True,
                        help="Directory containing per-class subdirectories of .npy files")
    parser.add_argument("--target", type=int, default=250,
                        help="Target number of samples per class (default: 250)")
    args = parser.parse_args()

    library_dir = Path(args.library_dir)
    if not library_dir.exists():
        raise FileNotFoundError(f"Library directory not found: {library_dir}")

    class_dirs = sorted([d for d in library_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"No class subdirectories found in {library_dir}")
        return

    print(f"\nAugmenting classes in {library_dir} → target {args.target}\n")
    total = 0
    for class_dir in class_dirs:
        generated = augment_class(class_dir, class_dir.name, args.target)
        total += generated

    print(f"\nDone. Generated {total} new augmented samples.")
    print("\nFinal counts:")
    for class_dir in class_dirs:
        count = len(list(class_dir.glob(f"{class_dir.name}_*.npy")))
        print(f"  {class_dir.name}: {count}")


if __name__ == "__main__":
    main()
