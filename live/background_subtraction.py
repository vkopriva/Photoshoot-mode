"""
Background subtraction for point clouds.
Isolates foreground objects from static environment.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement
from datetime import datetime
import os


def subtract_background(
    current: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    max_distance: float | None = None,
    min_z: float | None = None,
) -> np.ndarray:
    """
    Find points in current that don't exist in reference.

    Args:
        current: Current frame points (N, 3)
        reference: Reference/empty scene points (M, 3)
        threshold: Distance threshold in meters
        max_distance: Drop points farther than this from the sensor origin
        min_z: Drop points with z below this value (e.g. ground removal)

    Returns:
        Foreground points (K, 3)
    """
    tree = KDTree(reference)
    distances, _ = tree.query(current)

    foreground_mask = distances > threshold
    foreground = current[foreground_mask]
    if max_distance is not None:
        point_ranges = np.linalg.norm(foreground, axis=1)
        foreground = foreground[point_ranges <= max_distance]
    if min_z is not None:
        foreground = foreground[foreground[:, 2] >= min_z]
    return foreground

def load_reference_scene(path: str): #-> np.ndarray:
    """Load reference scene from numpy file."""
    data = PlyData.read(path)
    return np.array([[x, y, z] for x, y, z in zip(data.elements[0].data['x'], data.elements[0].data['y'], data.elements[0].data['z'])])

def load_current_scene(path: str): #-> np.ndarray:
    """Load current scene from PLY file."""
    data = PlyData.read(path)
    return np.array([[x, y, z] for x, y, z in zip(data.elements[0].data['x'], data.elements[0].data['y'], data.elements[0].data['z'])])

def save_reference_scene(points: np.ndarray, path: str) -> None:
    """Save reference scene to numpy file."""
    np.save(path, points)

def save_output_npy(points: np.ndarray, path: str) -> None:
    """Save foreground points to a numpy file."""
    np.save(path, points)

def save_output_ply(points: np.ndarray, path: str) -> None:
    """Save foreground points to a PLY file."""
    if points.size == 0:
        print("Warning: No foreground points to save.")
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
    now = datetime.now()

    parser = argparse.ArgumentParser(description="Define use of augmentation script")
    parser.add_argument("--dir", type=str, help="Directory of the input data. Requires [object_name]_scan[int].ply files and a single reference.ply")
    parser.add_argument("--library_dir", type=str, help="Directory of the object library")
    parser.add_argument("--visualize", action="store_true", help="Creates PLY files for visualisation in CloudCompare")
    parser.add_argument("--thresh", type=float, default=0.01, help="Threshold for background subtraction")
    parser.add_argument("--max_distance", type=float, default=None, help="Maximum distance from sensor to keep points (in meters)")
    parser.add_argument("--min_z", type=float, default=None, help="Drop foreground points with z below this value (in meters)")
    args = parser.parse_args()

    # BEST THRESHOLD = 0.01, NO FURTHER IMPROVEMENT BELOW
    threshold = args.thresh  # meters
    path_to_file = args.dir
    library_dir = args.library_dir
    class_name = Path(path_to_file).stem.rsplit('_', 1)[0]
    path_to_folder = Path(path_to_file).parent
    
    #check for the total number of scans in the folder, assuming they are named as [object_name]_[int].ply
    num_of_scans = max(
        int(scan.stem.rsplit('_', 1)[1])
        for scan in Path(path_to_folder).glob(f"{class_name}_*.ply")
    )
    print(f"Found {num_of_scans} scans for class {class_name} in {path_to_folder}.")
    # Build full file paths
    reference_path = os.path.join(
        path_to_folder,
        f"reference.ply"
    )

    print("Loading reference scene from:", reference_path)
    reference = load_reference_scene(reference_path)

    # build output library folder
    dest = os.path.join(library_dir, class_name)
    os.makedirs(dest, exist_ok=True)

    # build output library folder
    vis = os.path.join(Path(library_dir).parent, "visualization")
    os.makedirs(vis, exist_ok=True)

    for file in range (1, int(num_of_scans)+1):
        current_path = os.path.join(
        path_to_folder,
        f"{class_name}_{file}.ply"
        )

        output_path = os.path.join(
        dest,
        f"{class_name}_{file}.npy"
        )

        if os.path.exists(output_path):
            print(f"Warning: {output_path} already exists and will not be overwritten.")
            continue
        
        output_path_ply = os.path.join(
        vis,
        f"{class_name}_{file}.ply"
        )
        #+now.strftime("%Y-%m-%d_%H-%M")+"_thresh_"+str(threshold)

        print("Loading current frame from:", current_path)
        current = load_current_scene(current_path)

        foreground = subtract_background(
            current,
            reference,
            threshold,
            max_distance=args.max_distance,
            min_z=args.min_z,
        )

        print(f"Detected {len(foreground)} foreground points.")

        save_output_npy(foreground, output_path)
        if args.visualize:
            save_output_ply(foreground, output_path_ply)

    print("Done.")