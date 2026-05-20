#!/usr/bin/env python3
'''
Master file to run the live pipeline end-to-end:
background subtraction -> clustering -> normalization -> classification

All steps run in memory without intermediate file I/O.

BASED ON live_master2.py
'''
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import faiss

from background_subtraction import subtract_background
from live_background_subtraction import BinaryFrameProcessor
from live_classification_bb import ClusterProcessor
from dgcnn_light import DGCNN
from live_metrics import evaluate_summary, print_metrics

EMBED_DIM = 256
K_NEIGHBORS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MasterPipeline:
    def __init__(
        self,
        bg_threshold: float = 0.01,
        bg_max_distance: float = 5.0,
        bg_visualize: bool = False,
        bg_output_dir: Optional[Path] = None,
        save_foreground: bool = False,
        cluster_eps: float = 0.2,
        cluster_min_samples: int = 5,
        cluster_min_cluster_size: int = 40,
        cluster_max_cluster_size: Optional[int] = None,
        cluster_visualize: bool = False,
        cluster_save_fig: Optional[str] = None,
        cluster_point_size: float = 1.0,
        cluster_max_normalized_plots: int = 6,
        cluster_show_plots: bool = True,
        classification_assets_dir: Optional[str] = None,
        classification_threshold: float = 0.7,
        classification_k: int = 5,
        verbose: bool = False,
    ):
        self.bg_processor = BinaryFrameProcessor(
            output_dir=bg_output_dir,
            threshold=bg_threshold,
            max_distance=bg_max_distance,
            visualize=bg_visualize,
            save_foreground=save_foreground,
        )

        # Load classification assets if path is provided
        encoder = None
        index = None
        class_labels = None

        if classification_assets_dir:
            assets_dir = Path(classification_assets_dir)

            # Load encoder
            encoder_path = next(assets_dir.glob("*encoder*.pth"))
            encoder = DGCNN(embed_dim=EMBED_DIM, k=K_NEIGHBORS).to(DEVICE)
            state_dict = torch.load(encoder_path, map_location=DEVICE)
            state_dict = {k.removeprefix("encoder."): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict)
            encoder.eval()

            # Load FAISS index and labels
            index = faiss.read_index(str(assets_dir / "library.index"))
            labels_path = assets_dir / "labels.json"
            if labels_path.exists():
                class_labels = json.load(open(labels_path))

        self.cluster_processor = ClusterProcessor(
            eps=cluster_eps,
            min_samples=cluster_min_samples,
            min_cluster_size=cluster_min_cluster_size,
            max_cluster_size=cluster_max_cluster_size,
            visualize=cluster_visualize,
            save_fig=cluster_save_fig,
            point_size=cluster_point_size,
            max_normalized_plots=cluster_max_normalized_plots,
            show_plots=cluster_show_plots,
            encoder=encoder,
            index=index,
            class_labels=class_labels,
            threshold=classification_threshold,
            k=classification_k,
            verbose=verbose,
        )

        self.bg_visualize = bg_visualize
        self.bg_threshold = bg_threshold
        self.bg_max_distance = bg_max_distance
        self.verbose = verbose

    def close(self) -> None:
        self.bg_processor.close_visualizer()

    def run(
        self,
        input_scan: str,
        background_scan: str,
        start_frame: int = 0,
        max_frames: int = 100,
        reference_frame: int = 0,
    ) -> Dict[str, Any]:
        
        print("\n==========================================")
        print(" MASTER PIPELINE STARTED")
        print("==========================================")
        print(f"Input scan            : {input_scan}")
        print(f"Background scan       : {background_scan}")
        print(f"Start frame           : {start_frame}")
        print(f"Max frames            : {max_frames}")
        print(f"Reference frame       : {reference_frame}")
        print(f"Background threshold  : {self.bg_threshold}")
        print(f"Background max dist   : {self.bg_max_distance}")
        print(f"BG visualize          : {self.bg_processor.visualize}")
        print(f"Cluster visualize     : {self.cluster_processor.visualize}")
        print("==========================================")

        # Step 1: load reference cloud
        reference_points = self.bg_processor._extract_frame_points(
            pcap_path=background_scan,
            frame_idx=reference_frame,
        )
        print(f"[background] Reference cloud has {len(reference_points):,} points.")

        # Step 2: load foreground PCAP
        scan_source, sensor_info, xyz_lut = self.bg_processor._load_scan_source_and_lut(input_scan)

        self.bg_processor.native_resolution = (
            sensor_info.format.pixels_per_column,
            sensor_info.format.columns_per_frame,
        )

        scans = list(scan_source)
        if len(scans) == 0:
            raise RuntimeError(f"No scans found in foreground PCAP: {input_scan}")

        if start_frame < 0 or start_frame >= len(scans):
            raise ValueError(
                f"start_frame={start_frame} out of range for {len(scans)} scans"
            )

        end_frame = min(start_frame + max_frames, len(scans))

        results: List[Dict[str, Any]] = []
        total_clustered_points = 0
        total_noise_points = 0
        total_clusters = 0

        classification_counts: Dict[str, int] = {}

        for frame_idx in range(start_frame, end_frame):
            print(f"\n--- Frame {frame_idx} ---")

            scan_packet = scans[frame_idx]
            scan = scan_packet[0]

            if scan is None:
                print(f"[frame {frame_idx}] Missing scan, skipping.")
                continue

            current_points = self.bg_processor.scan_to_points(scan, xyz_lut)

            if current_points.size == 0:
                print(f"[frame {frame_idx}] Empty scan, skipping.")
                continue

            # Step 3: background subtraction in memory
            foreground = subtract_background(
                current=current_points,
                reference=reference_points,
                threshold=self.bg_processor.threshold,
                max_distance=self.bg_max_distance,
            )

            self.bg_processor._save_frame(foreground, frame_idx)

            print(
                f"[frame {frame_idx}] input={len(current_points):,} "
                f"foreground={len(foreground):,}"
            )

            self.bg_processor.stats["frames_processed"] += 1
            self.bg_processor.stats["total_input_points"] += len(current_points)
            self.bg_processor.stats["total_foreground_points"] += len(foreground)

            if self.bg_visualize:
                self.bg_processor._visualize_foreground(foreground, frame_idx)

            # Step 4: clustering + normalization + classification in memory
            cluster_result = self.cluster_processor.run(
                points=foreground,
                frame_idx=frame_idx,
            )

            n_noise = cluster_result["num_noise_points"]
            n_input = len(cluster_result["input_points"])
            n_clustered = n_input - n_noise
            n_clusters = cluster_result["num_clusters"]
            classifications = cluster_result.get("classifications", [])

            total_noise_points += n_noise
            total_clustered_points += n_clustered
            total_clusters += n_clusters

            print(f"[frame {frame_idx}] Classification results:")
            if classifications:
                for cid, (label, conf) in enumerate(classifications):
                    print(f"  Cluster {cid}: {label} (conf: {conf:.3f})")
                    classification_counts[label] = classification_counts.get(label, 0) + 1
            else:
                print("  No classifications returned.")

            results.append(
                {
                    "frame_idx": frame_idx,
                    "input_points": len(current_points),
                    "foreground_points": len(foreground),
                    "cluster_input_points": n_input,
                    "clustered_points": n_clustered,
                    "noise_points": n_noise,
                    "num_clusters": n_clusters,
                    "labels": cluster_result["labels"],
                    "raw_clusters": cluster_result["raw_clusters"],
                    "processed_clusters": cluster_result["processed_clusters"],
                    "clusters": cluster_result["clusters"],
                    "normalized_clusters": cluster_result["normalized_clusters"],
                    "bounding_boxes": cluster_result["bounding_boxes"],
                    "classifications": classifications,
                }
            )

        summary = {
            "input_scan": str(input_scan),
            "background_scan": str(background_scan),
            "reference_frame": reference_frame,
            "frames_processed": self.bg_processor.stats["frames_processed"],
            "total_input_points": self.bg_processor.stats["total_input_points"],
            "total_foreground_points": self.bg_processor.stats["total_foreground_points"],
            "total_clustered_points": total_clustered_points,
            "total_noise_points": total_noise_points,
            "total_clusters_detected": total_clusters,
            "classification_counts": classification_counts,
            "frame_results": results,
        }

        print("\n==========================================")
        print(" MASTER PIPELINE COMPLETE")
        print("==========================================")
        print(f"Frames processed       : {summary['frames_processed']}")
        print(f"Total input points     : {summary['total_input_points']:,}")
        print(f"Total foreground points: {summary['total_foreground_points']:,}")
        print(f"Total clustered points : {summary['total_clustered_points']:,}")
        print(f"Total noise points     : {summary['total_noise_points']:,}")
        print(f"Total clusters         : {summary['total_clusters_detected']}")
        print("Classification counts  :")
        if classification_counts:
            for label, count in sorted(classification_counts.items()):
                print(f"  {label}: {count}")
        else:
            print("  None")
        print("==========================================")

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master pipeline: background subtraction -> clustering -> normalization -> classification"
    )

    # Input PCAPs
    parser.add_argument("input_scan", type=str, help="Path to foreground/input Ouster .pcap file")
    parser.add_argument("background", type=str, help="Path to background/reference Ouster .pcap file")

    # Background subtraction params
    parser.add_argument("--start", type=int, default=0, help="First frame to process from the foreground PCAP")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process from the foreground PCAP")
    parser.add_argument("--reference-frame", type=int, default=0, help="Frame index to use from the background PCAP")
    parser.add_argument("--knn_threshold", type=float, default=0.01, help="Threshold for nearest-neighbor background subtraction")
    parser.add_argument("--max_distance", type=float, default=5.0, help="Maximum distance from sensor to keep points in meters")
    parser.add_argument("--visualize-bg", action="store_true", help="Visualize foreground in Open3D")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save per-frame .npy foreground files")
    parser.add_argument("--save-foreground", action="store_true", help="Save per-frame foreground .npy files")

    # Clustering params
    parser.add_argument("--visualize-clusters", action="store_true", help="Visualize clusters using matplotlib")
    parser.add_argument("--eps", type=float, default=0.2, help="DBSCAN neighbourhood radius in metres")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum points to form a cluster core")
    parser.add_argument("--min_cluster_size", type=int, default=40, help="Minimum points for a valid cluster")
    parser.add_argument("--max_cluster_size", type=int, default=None, help="Maximum points for a valid cluster")
    parser.add_argument("--cluster-save-fig", type=str, default=None, help="Optional output path prefix for saved cluster figures")
    parser.add_argument("--cluster-point-size", type=float, default=1.0, help="Matplotlib scatter marker size")
    parser.add_argument("--max-normalized-plots", type=int, default=6, help="Maximum number of normalized cluster subplots to show")

    # Classification params
    parser.add_argument("--assets_dir", type=str, required=True, help="Directory containing encoder .pth, library.index, and labels.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum confidence for predicted class")
    parser.add_argument("--k", type=int, default=5, help="Top-k FAISS neighbors for voting")
    parser.add_argument("--verbose", action="store_true", help="Show detailed clustering/classification output per frame")

    # Metrics params
    parser.add_argument("--metrics-ground-truth", type=str, default=None, help="Optional ground-truth JSON/CSV file for 3D IoU, mAP, precision, and recall")
    parser.add_argument("--metrics-iou-threshold", type=float, default=0.5, help="Minimum 3D IoU used to count a true positive")
    parser.add_argument("--metrics-class-agnostic", action="store_true", help="Evaluate detected boxes without comparing class labels")
    parser.add_argument("--metrics-output", type=str, default=None, help="Optional path to save metrics as JSON")

    args = parser.parse_args()

    pipeline = MasterPipeline(
        bg_threshold=args.knn_threshold,
        bg_max_distance=args.max_distance,
        bg_visualize=args.visualize_bg,
        bg_output_dir=Path(args.output_dir) if args.output_dir else None,
        save_foreground=args.save_foreground,
        cluster_eps=args.eps,
        cluster_min_samples=args.min_samples,
        cluster_min_cluster_size=args.min_cluster_size,
        cluster_max_cluster_size=args.max_cluster_size,
        cluster_visualize=args.visualize_clusters,
        cluster_save_fig=args.cluster_save_fig,
        cluster_point_size=args.cluster_point_size,
        cluster_max_normalized_plots=args.max_normalized_plots,
        cluster_show_plots=True,
        classification_assets_dir=args.assets_dir,
        classification_threshold=args.threshold,
        classification_k=args.k,
        verbose=args.verbose,
    )

    try:
        summary = pipeline.run(
            input_scan=args.input_scan,
            background_scan=args.background,
            start_frame=args.start,
            max_frames=args.frames,
            reference_frame=args.reference_frame,
        )
        print("\nDone.")
        print(
            {
                "frames_processed": summary["frames_processed"],
                "total_input_points": summary["total_input_points"],
                "total_foreground_points": summary["total_foreground_points"],
                "total_clustered_points": summary["total_clustered_points"],
                "total_noise_points": summary["total_noise_points"],
                "total_clusters_detected": summary["total_clusters_detected"],
                "classification_counts": summary["classification_counts"],
            }
        )

        if args.metrics_ground_truth:
            metrics = evaluate_summary(
                summary=summary,
                ground_truth_path=args.metrics_ground_truth,
                iou_threshold=args.metrics_iou_threshold,
                class_agnostic=args.metrics_class_agnostic,
            )
            print_metrics(metrics)

            if args.metrics_output:
                with open(args.metrics_output, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"[metrics] Saved metrics to {args.metrics_output}")

        if args.visualize_bg:
            while pipeline.bg_processor.vis is not None:
                pipeline.bg_processor.vis.poll_events()
                pipeline.bg_processor.vis.update_renderer()

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
