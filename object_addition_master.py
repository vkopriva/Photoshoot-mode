"""
generate_training_data.py
==================

End-to-end object addition pipeline:
    .ply scans → background subtraction → clustering → review → augmentation → FAISS index

"""

import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from background_subtraction import (
    load_reference_scene,
    load_current_scene,
    subtract_background,
    cluster_foreground,
    save_output_npy,
    save_output_ply,
)


def augment_point_cloud(pts: np.ndarray) -> np.ndarray:
    """Apply realistic LiDAR augmentations to a normalized point cloud."""
    pts = pts.copy()

    # Small Z-axis rotation (±30°) — sensor sees object from similar angles
    theta = np.random.uniform(-np.pi / 6, np.pi / 6)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T

    # Random scaling (0.85–1.15) — objects at different distances
    scale = np.random.uniform(0.85, 1.15)
    pts *= scale

    # Small translation jitter — imperfect centering from background subtraction
    pts += np.random.uniform(-0.05, 0.05, size=(1, 3)).astype(np.float32)

    # Gaussian jitter (LiDAR measurement noise)
    pts += np.random.normal(0, 0.005, pts.shape).astype(np.float32)

    MIN_POINTS = 30

    # Random point dropout (20–40%) — occlusion and partial views
    # Only apply if we won't go below MIN_POINTS
    drop_rate = np.random.uniform(0.2, 0.4)
    target_n = max(int(len(pts) * (1 - drop_rate)), MIN_POINTS)
    if target_n < len(pts):
        keep = np.random.choice(len(pts), target_n, replace=False)
        pts = pts[keep]

    # Density variation — randomly subsample to simulate near/far scans
    # Only apply if we won't go below MIN_POINTS
    if random.random() < 0.5 and len(pts) > MIN_POINTS:
        sparse_n = max(int(len(pts) * np.random.uniform(0.5, 0.8)), MIN_POINTS)
        if sparse_n < len(pts):
            idx = np.random.choice(len(pts), sparse_n, replace=False)
            pts = pts[idx]

    # Re-normalize to unit sphere after augmentations
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1).max()
    if d > 0:
        pts /= d

    return pts.astype(np.float32)


def open_point_cloud_with_open3d(points: np.ndarray, title: str, color=(1.0, 0.0, 0.0)) -> None:
    """Show a single point cloud in Open3D."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    print("  Close the Open3D window to continue.")
    o3d.visualization.draw_geometries([cloud, frame], window_name=title)


def show_point_cloud_and_prompt(points: np.ndarray, title: str, tmp_path: Path | None = None, review_png: bool = False) -> bool:
    """Show a single point cloud and prompt y/n."""
    if not review_png:
        open_point_cloud_with_open3d(points, title)
        while True:
            answer = input(f"  Accept this augmentation? [y/n]: ").strip().lower()
            if answer in ("y", "n"):
                return answer == "y"
            print("  Please enter 'y' or 'n'.")

    if tmp_path is None:
        raise ValueError("tmp_path is required when review_png=True")

    import platform

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=3, c="red", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}  ({len(points)} points)")

    ranges = points.max(axis=0) - points.min(axis=0)
    max_range = ranges.max() / 2
    mid = (points.max(axis=0) + points.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig(tmp_path, dpi=130, bbox_inches="tight")
    plt.close()

    if platform.system() == "Linux":
        subprocess.Popen(["xdg-open", str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(tmp_path)])
    else:
        subprocess.Popen(["start", str(tmp_path)], shell=True)

    while True:
        answer = input(f"  Accept this augmentation? [y/n]: ").strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("  Please enter 'y' or 'n'.")


def normalize_points(pts: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere."""
    pts = pts - pts.mean(axis=0)
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts /= d
    return pts.astype(np.float32)


def open_clusters_with_open3d(clusters, title: str) -> None:
    """Show colored foreground clusters in Open3D."""
    import open3d as o3d

    cmap = plt.get_cmap("tab20")
    geometries = []

    print("  Clusters:")
    for i, (_, pts) in enumerate(clusters):
        rgb = [float(c) for c in cmap(i % 20)[:3]]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.paint_uniform_color(rgb)
        geometries.append(cloud)
        print(f"    {i}: {len(pts)} points, color RGB={tuple(round(c, 2) for c in rgb)}")

    all_pts = np.vstack([pts for _, pts in clusters])
    span = all_pts.max(axis=0) - all_pts.min(axis=0)
    frame_size = max(float(span.max()) * 0.15, 0.1)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    geometries.append(frame)

    print("  Close the Open3D window to pick a cluster in the terminal.")
    o3d.visualization.draw_geometries(geometries, window_name=title)


def show_clusters_and_prompt(clusters, title: str, tmp_path: Path | None = None, review_png: bool = False):
    """Show colored clusters and prompt user to pick one."""
    if not review_png:
        open_clusters_with_open3d(clusters, title)
        valid_choices = [str(i) for i in range(len(clusters))] + ["n"]
        while True:
            answer = input(f"  Pick cluster [0-{len(clusters)-1}] or 'n' to skip: ").strip().lower()
            if answer in valid_choices:
                if answer == "n":
                    return None
                return clusters[int(answer)][1]
            print(f"  Please enter a number 0-{len(clusters)-1} or 'n'.")

    if tmp_path is None:
        raise ValueError("tmp_path is required when review_png=True")

    import platform

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(20)]
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_side = fig.add_subplot(2, 2, 2)
    ax_front = fig.add_subplot(2, 2, 4)

    all_pts = np.vstack([pts for _, pts in clusters])

    for i, (lbl, pts) in enumerate(clusters):
        color = colors[i % len(colors)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=3, color=color, alpha=0.8, label=f"Cluster {i} ({len(pts)} pts)")
        # Side view: looking along X (lateral), show Y (forward) vs Z (up)
        ax_side.scatter(pts[:, 1], pts[:, 2], s=3, color=color, alpha=0.8,
                        label=f"Cluster {i} ({len(pts)} pts)")
        # Front view: looking along Y (forward), show X (lateral) vs Z (up)
        ax_front.scatter(pts[:, 0], pts[:, 2], s=3, color=color, alpha=0.8,
                         label=f"Cluster {i} ({len(pts)} pts)")

    ax.set_xlabel("X (lateral)")
    ax.set_ylabel("Y (forward)")
    ax.set_zlabel("Z (up)")
    ax.set_title(f"{title} — 3D view")
    ax.legend(loc="upper left", fontsize=8, markerscale=3)

    ax_side.set_xlabel("Y (forward)")
    ax_side.set_ylabel("Z (up)")
    ax_side.set_title("Side view (Y-Z plane, looking along X)")
    ax_side.set_aspect("equal")
    ax_side.grid(True, alpha=0.3)
    ax_side.legend(loc="upper left", fontsize=7, markerscale=3)

    ax_front.set_xlabel("X (lateral)")
    ax_front.set_ylabel("Z (up)")
    ax_front.set_title("Front view (X-Z plane, looking along Y)")
    ax_front.set_aspect("equal")
    ax_front.grid(True, alpha=0.3)
    ax_front.legend(loc="upper left", fontsize=7, markerscale=3)

    # Equal aspect ratio for 3D plot
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = ranges.max() / 2
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig(tmp_path, dpi=130, bbox_inches="tight")
    plt.close()

    # Open the image
    if platform.system() == "Linux":
        subprocess.Popen(["xdg-open", str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(tmp_path)])
    else:
        subprocess.Popen(["start", str(tmp_path)], shell=True)

    # Prompt user
    valid_choices = [str(i) for i in range(len(clusters))] + ["n"]
    while True:
        answer = input(f"  Pick cluster [0-{len(clusters)-1}] or 'n' to skip: ").strip().lower()
        if answer in valid_choices:
            if answer == "n":
                return None
            return clusters[int(answer)][1]
        print(f"  Please enter a number 0-{len(clusters)-1} or 'n'.")


def run_background_subtraction(parent_dir, class_name, library_dir, reference, threshold, max_distance, floor_z=None, review=False, review_png=False, eps=0.3):
    print("\n[1/3] Running background subtraction...", flush=True)

    parent_dir = Path(parent_dir)
    library_dir = Path(library_dir)
    reference = Path(reference)

    # Load reference (empty background) scene once
    ref_folder = reference.parent.name
    print(f"Loading reference scene from: {reference}")
    ref_points = load_reference_scene(str(reference))

    # Floor removal — drop points below floor_z threshold
    if floor_z is not None:
        before = len(ref_points)
        ref_points = ref_points[ref_points[:, 2] > floor_z]
        print(f"  Floor filter (Z > {floor_z}): {before} → {len(ref_points)} reference points")

    # Output directory
    dest = library_dir / class_name
    dest.mkdir(parents=True, exist_ok=True)

    # Iterate over subfolders, skipping the reference folder
    # Only process folders whose .ply files match the class name
    scan_dirs = sorted([d for d in parent_dir.iterdir() if d.is_dir() and d.name != ref_folder])
    scan_dirs = [d for d in scan_dirs if any(f.stem.startswith(class_name) for f in d.glob("*.ply"))]
    print(f"Found {len(scan_dirs)} scan folders for class '{class_name}' (excluding reference: {ref_folder})")

    count = 0
    saved = 0
    for scan_dir in scan_dirs:
        # Find .ply files in each subfolder
        ply_files = sorted(scan_dir.glob("*.ply"))
        if not ply_files:
            continue

        for ply_file in ply_files:
            count += 1
            output_path = dest / f"{class_name}_{count}.npy"

            if output_path.exists():
                print(f"  Skipping (exists): {output_path}")
                continue

            print(f"  Processing: {ply_file}")
            current = load_current_scene(str(ply_file))
            if floor_z is not None:
                current = current[current[:, 2] > floor_z]
            foreground = subtract_background(current, ref_points, threshold, max_distance=max_distance)
            print(f"    → {len(foreground)} foreground points")

            if len(foreground) == 0:
                print(f"    Warning: no foreground points, skipping")
                count -= 1
                continue

            clusters = cluster_foreground(foreground, eps=eps, min_cluster_size=30)

            if not clusters:
                print(f"    Warning: no clusters with >= 30 points, skipping")
                count -= 1
                continue

            if review:
                tmp_img = dest / f".preview_{class_name}_{count}.png" if review_png else None
                selected = show_clusters_and_prompt(
                    clusters,
                    title=f"{class_name}_{count} ({ply_file.parent.name})",
                    tmp_path=tmp_img,
                    review_png=review_png,
                )
                if tmp_img is not None:
                    tmp_img.unlink(missing_ok=True)

                if selected is None:
                    print(f"    ✗ Skipped")
                    count -= 1
                    continue
                foreground = selected
                print(f"    ✓ Accepted ({len(foreground)} points)")
            else:
                # Without review, take the largest cluster
                foreground = max(clusters, key=lambda c: len(c[1]))[1]

            foreground = normalize_points(foreground)
            save_output_npy(foreground, str(output_path))
            saved += 1

    print(f"\n  Saved {saved} samples to {dest}")


def run_augmentation(class_dir, num_aug):
    print("\n[2/3] Running augmentation...", flush=True)

    cmd = [
        sys.executable,
        "augmentation.py",
        "--dir", str(class_dir),
        "--num-aug", str(num_aug),
    ]

    subprocess.run(cmd, check=True)


def run_faiss_expansion(class_dir, assets_dir, class_name):
    print("\n[3/3] Building FAISS index...", flush=True)

    cmd = [
        sys.executable,
        "expand_faiss.py",
        "--data_dir", str(class_dir),
        "--assets_dir", str(assets_dir),
        "--class_name", str(class_name),
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Full object addition pipeline")

    # ── Required inputs ───────────────────────────────────────────────────────
    parser.add_argument("--parent_dir", type=str, required=True, 
                        help="Parent directory containing per-scene subfolders")
    parser.add_argument("--class_name", type=str, required=True, 
                        help="Class prefix to extract, e.g. chair")
    parser.add_argument("--library_dir", required=True,
                        help="Directory where processed objects are stored")
    parser.add_argument("--assets_dir", required=True,
                        help="Directory containing encoder weights and FAISS outputs")
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to the background/reference PLY file")

    # ── Pipeline params ───────────────────────────────────────────────────────
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Background subtraction threshold")
    parser.add_argument("--max_distance", type=float, default=5,
                        help="Maximum distance for points to be considered foreground")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="DBSCAN eps parameter for clustering")
    parser.add_argument("--num_aug", type=int, default=5,
                        help="Number of augmentations per sample")
    parser.add_argument("--floor_z", type=float, default=-0.87,
                        help="Drop points with Z below this value (floor removal). Set to None to disable.")
    parser.add_argument("--review", action="store_true",
                        help="Visually review each point cloud before adding to FAISS index")
    parser.add_argument("--review-png", action="store_true",
                        help="Use temporary PNG files for review instead of Open3D windows")

    args = parser.parse_args()
    class_name = args.class_name
    parent_dir = Path(args.parent_dir)
    library_dir = Path(args.library_dir)
    assets_dir = Path(args.assets_dir)
    review = args.review
    review_png = args.review_png
    review = review or review_png  # Enable review if review_png is set

    # ── Validate input ────────────────────────────────────────────────────────
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")
    if not any(assets_dir.glob("*encoder*.pth")):
        raise FileNotFoundError(f"No encoder weights matching '*encoder*.pth' found in: {assets_dir}")

    class_name = args.class_name
    class_dir = library_dir / class_name

    print("\n==========================================")
    print(" DATA GENERATION STARTED")
    print("==========================================")
    print(f"Class           : {class_name}")
    print(f"Parent dir      : {parent_dir}")
    print(f"Library dir     : {library_dir}")
    print(f"Assets dir      : {assets_dir}")
    print("==========================================", flush=True)

    # ── Step 1: Background subtraction ────────────────────────────────────────
    run_background_subtraction(
        parent_dir=parent_dir,
        class_name=class_name,
        library_dir=library_dir,
        reference=args.reference,
        threshold=args.threshold,
        max_distance=args.max_distance,
        floor_z=args.floor_z,
        review=review,
        review_png=review_png,
        eps=args.eps,
    )

    # ── Step 2: Augmentation ──────────────────────────────────────────────────
    if not class_dir.exists():
        raise RuntimeError(f"Expected class directory not found: {class_dir}")

    num_aug = args.num_aug
    print(f"\n[2/3] Generating {num_aug} augmentations per sample...")

    base_files = sorted(class_dir.glob(f"{class_name}_*.npy"))
    aug_count = 0
    # Start numbering after existing files
    next_id = len(base_files) + 1

    for base_file in base_files:
        base_pts = np.load(base_file).astype(np.float32)
        print(f"\n  Augmenting: {base_file.name}")

        # Review only the first augmentation. If accepted, generate the rest
        # automatically. If rejected, skip all augmentations for this sample.
        if review:
            first_aug = augment_point_cloud(base_pts)
            tmp_img = class_dir / f".preview_aug_{next_id}.png" if review_png else None
            accepted = show_point_cloud_and_prompt(
                first_aug,
                title=f"{base_file.stem} → aug 1/{num_aug} (preview)",
                tmp_path=tmp_img,
                review_png=review_png,
            )
            if tmp_img is not None:
                tmp_img.unlink(missing_ok=True)

            if not accepted:
                print(f"    ✗ Rejected — skipping all augmentations for {base_file.name}")
                continue

            print(f"    ✓ Accepted — generating {num_aug} augmentations")
            save_output_npy(first_aug, str(class_dir / f"{class_name}_{next_id}.npy"))
            next_id += 1
            aug_count += 1

            for a in range(1, num_aug):
                aug_pts = augment_point_cloud(base_pts)
                save_output_npy(aug_pts, str(class_dir / f"{class_name}_{next_id}.npy"))
                next_id += 1
                aug_count += 1
        else:
            for a in range(num_aug):
                aug_pts = augment_point_cloud(base_pts)
                save_output_npy(aug_pts, str(class_dir / f"{class_name}_{next_id}.npy"))
                next_id += 1
                aug_count += 1

    print(f"\n  Generated {aug_count} augmented samples in {class_dir}")

    # ── Step 3: FAISS index build ─────────────────────────────────────────────
    run_faiss_expansion(
        class_dir=class_dir,
        assets_dir=assets_dir,
        class_name=class_name,
    )

    print("\n==========================================")
    print(" OBJECT ADDITION FINISHED")
    print("==========================================")


if __name__ == "__main__":
    main()
