"""
expand_faiss.py

Usage:
    python build_faiss_index.py \
        --data_dir /path/to/npy_files \
        --assets_dir /path/to/assets_folder

Assumptions:
- data_dir contains .npy files (each = point cloud Nx3)
- assets_dir contains:
    - encoder weights (*encoder*.pth)
    - class_names.json
"""

import sys
import json
import argparse
import numpy as np
import torch
import faiss
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from dgcnn_light import DGCNN


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_POINTS    = 1024
EMBED_DIM   = 256
K_NEIGHBORS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts /= d
    return pts.astype(np.float32)


def load_npy(path: Path, n_points: int = 1024) -> np.ndarray:
    pts = np.load(path).astype(np.float32)

    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")
    # UPSAMPLING/DOWNSAMPLING TO FIXED N_POINTS (1024)
    # If fewer points, sample with replacement; if more, sample without replacement
    # This ensures consistent input size for the encoder regardless of original point count
    N = pts.shape[0]
    idx = np.random.choice(N, n_points, replace=(N < n_points))
    pts = pts[idx]

    return normalize_points(pts)


def resolve_assets(assets_dir: Path):
    encoder_path = next(assets_dir.glob("*encoder*.pth"))
    return encoder_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--class_name", required=True, help="Class label for all files in data_dir")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assets_dir = Path(args.assets_dir)
    class_name = args.class_name

    index_path = assets_dir / "library.index"
    labels_path = assets_dir / "labels.json"
    embeddings_path = assets_dir / "embeddings.npy"
    processed_path = assets_dir / "processed_files.json"

    np.random.seed(42)
    torch.manual_seed(42)

    # ── Resolve assets ────────────────────────────────────────────────────────
    encoder_path = resolve_assets(assets_dir)

    # ── Load encoder ──────────────────────────────────────────────────────────
    encoder = DGCNN(embed_dim=EMBED_DIM, k=K_NEIGHBORS).to(DEVICE)
    state_dict = torch.load(encoder_path, map_location=DEVICE)
    # Strip "encoder." prefix if the checkpoint was saved from a wrapper module
    state_dict = {k.removeprefix("encoder."): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict)
    encoder.eval()

    print("Encoder loaded.")

    # ── Load existing index if it exists ──────────────────────────────────────
    if index_path.exists():
        print("Loading existing FAISS index...")
        index = faiss.read_index(str(index_path))

        existing_labels = json.load(open(labels_path))
        existing_embeddings = np.load(embeddings_path)

        if processed_path.exists():
            processed_files = set(json.load(open(processed_path)))
        else:
            processed_files = set()

        print(f"Existing vectors: {index.ntotal}")
    else:
        print("Creating new FAISS index...")
        index = faiss.IndexFlatL2(EMBED_DIM)

        existing_labels = []
        existing_embeddings = np.empty((0, EMBED_DIM), dtype=np.float32)
        processed_files = set()

    # ── Load new files ────────────────────────────────────────────────────────
    files = sorted(data_dir.glob("*.npy"))

    new_embeddings = []
    new_labels = []
    new_files_added = []

    print("\nProcessing new files...\n")

    for path in files:
        if path.name in processed_files:
            continue  # skip already processed files

        # Load raw points (before normalization) for visualization
        raw_pts = np.load(path).astype(np.float32)

        pts = load_npy(path, N_POINTS)
        pts_t = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = encoder(pts_t).cpu().numpy()

        new_embeddings.append(emb.squeeze(0))

        new_labels.append(class_name)

        new_files_added.append(path.name)

    if len(new_embeddings) == 0:
        print("No new files to add. Index unchanged.")
        return

    new_embeddings = np.vstack(new_embeddings).astype(np.float32)

    print(f"Adding {len(new_embeddings)} new embeddings...")

    # ── Update index ──────────────────────────────────────────────────────────
    index.add(new_embeddings)

    # ── Merge data ────────────────────────────────────────────────────────────
    updated_embeddings = np.vstack([existing_embeddings, new_embeddings])
    updated_labels = existing_labels + new_labels
    processed_files.update(new_files_added)

    # ── Save everything ───────────────────────────────────────────────────────
    faiss.write_index(index, str(index_path))

    np.save(embeddings_path, updated_embeddings)

    with open(labels_path, "w") as f:
        json.dump(updated_labels, f, indent=2)

    with open(processed_path, "w") as f:
        json.dump(sorted(list(processed_files)), f, indent=2)

    print("\nIndex updated successfully")
    print(f"Total vectors: {index.ntotal}")
    print(f"New files added: {len(new_files_added)}")


if __name__ == "__main__":
    main()