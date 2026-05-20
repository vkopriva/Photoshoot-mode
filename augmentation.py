"""
Point cloud augmentation for training.
"""

import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import os
import sys


def augment_points(points: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to point cloud.

    Args:
        points: Input points (N, 3)

    Returns:
        Augmented points (N, 3)
    """
    points = points.copy()

    # Random rotation around Z-axis
    points = random_rotate_z(points)

    # Random jitter
    points = random_jitter(points)

    # Random scale
    points = random_scale(points)

    return points


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    """Rotate around Z-axis by random angle."""
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rotation = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    return points @ rotation.T


def random_rotate_full(points: np.ndarray) -> np.ndarray:
    """Random rotation in all axes (for non-upright objects)."""
    angles = np.random.uniform(0, 2 * np.pi, 3)

    # Rotation matrices for each axis
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    rotation = Rz @ Ry @ Rx
    return points @ rotation.T


def random_jitter(
    points: np.ndarray,
    sigma: float = 0.01,
    clip: float = 0.05
) -> np.ndarray:
    """Add random noise to point positions."""
    noise = np.clip(
        np.random.normal(0, sigma, points.shape),
        -clip, clip
    )
    return points + noise


def random_scale(
    points: np.ndarray,
    scale_range: tuple = (0.6, 1.5)
) -> np.ndarray:
    """Random uniform scaling."""
    scale = np.random.uniform(*scale_range)
    return points * scale


def random_dropout(
    points: np.ndarray,
    max_dropout: float = 0.2
) -> np.ndarray:
    """Randomly drop points."""
    dropout_ratio = np.random.uniform(0, max_dropout)
    n_keep = int(len(points) * (1 - dropout_ratio))
    indices = np.random.choice(len(points), n_keep, replace=False)
    return points[indices]


def center_points(points: np.ndarray) -> np.ndarray:
    """Center point cloud at origin."""
    centroid = points.mean(axis=0)
    return points - centroid


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize to unit sphere."""
    points = center_points(points)
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points

def load_model(path: str): #-> np.ndarray:
    data = np.load(path)
    return data

def save_output_npy(points: np.ndarray, path: str) -> None:
    """Save foreground points to a numpy file."""
    np.save(path, points)

def save_output_ply(points: np.ndarray, path: str) -> None:
    """Save foreground points to a PLY file."""
    if points.size == 0:
        print("Warning: No points to save.")
    # Ensure correct dtype (PLY expects structured array)
    vertex_data = np.array(
        [(x, y, z) for x, y, z in points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Define use of augmentation script")
    parser.add_argument("--dir", type=str, help="Directory of the input model. Program will save output in the same directory with suffixes _augmented_X.npy and _augmented_X.ply")
    parser.add_argument("--visualize", action="store_true", help="Creates PLY files for visualisation in CloudCompare")
    parser.add_argument("--num-aug", type=int, default=5, help="Number of augmentations to apply")
    args = parser.parse_args()
    
    reference_path = args.dir
    folder = Path(reference_path)
    class_name = Path(args.dir).stem    
    nr_of_augmentations = args.num_aug

    original_files = [
    f for f in folder.iterdir()
    if f.is_file()
    and f.suffix == ".npy"
    and "_augmented_" not in f.stem
]

    for i in range(nr_of_augmentations):
        for file in original_files:
            if file.is_file():
                points = load_model(os.path.join(reference_path, file.name))
                print("Loaded file:", file.name,"containing", points.shape,"points")
                augmented = augment_points(points)
                name = file.name.rsplit('.', 1)[0]
                output_path = os.path.join(folder, f"{name}_augmented_{i}.npy")
                if os.path.exists(output_path):
                    print(f"Warning: {output_path} already exists and will not be overwritten.")
                    continue
                save_output_npy(augmented, output_path)
                if args.visualize:
                    visualize_path = (Path(reference_path).parent).parent / "visualization"
                    if visualize_path.exists() and visualize_path.is_dir():
                        saved_path = visualize_path
                        save_output_ply(augmented, os.path.join(visualize_path, f"{name}_augmented_{i}.ply"))
                    else:
                        print(f"'visualize' folder not found in {Path(reference_path).parent}, visualizations not saved")
                print("Saved augmented shape:", augmented.shape)