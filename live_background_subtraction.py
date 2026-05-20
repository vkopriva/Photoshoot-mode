#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import open3d as o3d
from open3d.visualization import Visualizer

from ouster.sdk.core.data import ChanField, XYZLut
from ouster.sdk.open_source import open_source

from background_subtraction import subtract_background


class BinaryFrameProcessor:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        threshold: float = 0.01,
        max_distance: float = 10,
        visualize: bool = True,
        save_foreground: bool = False,
    ):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.threshold = threshold
        self.max_distance = max_distance
        self.visualize = visualize
        self.save_foreground = save_foreground

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.native_resolution = None

        self.stats = {
            "frames_processed": 0,
            "total_input_points": 0,
            "total_foreground_points": 0,
        }

        self.vis = None
        self.vis_initialized = False

    def scan_to_points(self, scan, xyz_lut) -> np.ndarray:
        xyz = xyz_lut(scan)
        range_data = scan.field(ChanField.RANGE)

        valid_mask = range_data > 0
        points = xyz[valid_mask]

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        return points.astype(np.float32)

    def _visualize_foreground(self, points: np.ndarray, frame_idx: int) -> None:
        if not self.visualize:
            return

        if points.size == 0:
            print(f"[frame {frame_idx}] No foreground points to visualize.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = np.zeros((points.shape[0], 3), dtype=np.float64)
        colors[:, 0] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.vis is None:
            self.vis = Visualizer()
            self.vis.create_window(window_name="Foreground", width=1280, height=720)
            self.vis.add_geometry(pcd)
        else:
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close_visualizer(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

    def _load_scan_source_and_lut(self, pcap_path: str):
        scan_source = open_source(str(pcap_path), sensor_idx=0)

        if not scan_source.sensor_info:
            raise RuntimeError(f"No sensor_info found in PCAP source: {pcap_path}")

        sensor_info = scan_source.sensor_info[0]
        xyz_lut = XYZLut(sensor_info)

        return scan_source, sensor_info, xyz_lut

    def _extract_frame_points(
        self,
        pcap_path: str,
        frame_idx: int,
    ) -> np.ndarray:
        scan_source, sensor_info, xyz_lut = self._load_scan_source_and_lut(pcap_path)

        scans = list(scan_source)

        if len(scans) == 0:
            raise RuntimeError(f"No scans found in PCAP: {pcap_path}")

        if frame_idx < 0 or frame_idx >= len(scans):
            raise ValueError(
                f"frame_idx={frame_idx} out of range for {len(scans)} scans in {pcap_path}"
            )

        scan_packet = scans[frame_idx]
        scan = scan_packet[0]

        if scan is None:
            raise RuntimeError(
                f"Frame {frame_idx} in {pcap_path} does not contain a valid scan."
            )

        self.native_resolution = (
            sensor_info.format.pixels_per_column,
            sensor_info.format.columns_per_frame,
        )

        return self.scan_to_points(scan, xyz_lut)
    
    def _save_frame(self, points: np.ndarray, frame_idx: int) -> None:
        if not self.save_foreground or self.output_dir is None:
            return
        filename = self.output_dir / f"frame_{frame_idx:06d}.npy"
        np.save(filename, points)

    def process_pcap(
        self,
        pcap_path: str,
        background_pcap_path: str,
        max_frames: int = 100,
        start_frame: int = 0,
        reference_frame: int = 0,
        max_distance: int = 5,
        visualize: bool = False,
        return_frames: bool = False,
    ):
        
        print(f"\nProcessing foreground PCAP: {pcap_path}")
        print(f"Using background PCAP: {background_pcap_path}")
        print(f"Start frame: {start_frame}")
        print(f"Max frames: {max_frames}")
        print(f"Background reference frame: {reference_frame}")
        print(f"kNN Threshold: {self.threshold}")
        print(f"Max distance: {max_distance}")

        frames = [] if return_frames else None

        reference_points = self._extract_frame_points(
            pcap_path=background_pcap_path,
            frame_idx=reference_frame,
        )

        print(f"Background reference cloud has {len(reference_points)} valid points.")

        scan_source, sensor_info, xyz_lut = self._load_scan_source_and_lut(pcap_path)

        self.native_resolution = (
            sensor_info.format.pixels_per_column,
            sensor_info.format.columns_per_frame,
        )

        scans = list(scan_source)

        if len(scans) == 0:
            raise RuntimeError(f"No scans found in foreground PCAP: {pcap_path}")

        if start_frame < 0 or start_frame >= len(scans):
            raise ValueError(
                f"start_frame={start_frame} out of range for {len(scans)} scans"
            )

        end_frame = min(start_frame + max_frames, len(scans))

        for frame_idx in range(start_frame, end_frame):
            scan_packet = scans[frame_idx]
            scan = scan_packet[0]

            if scan is None:
                print(f"[frame {frame_idx}] Missing scan, skipping.")
                continue

            current_points = self.scan_to_points(scan, xyz_lut)

            if current_points.size == 0:
                print(f"[frame {frame_idx}] Empty scan, skipping.")
                continue

            foreground = subtract_background(
                current=current_points,
                reference=reference_points,
                threshold=self.threshold,
                max_distance=max_distance,
            )

            if return_frames:
                frames.append({"frame_idx": frame_idx,"points": foreground})

            self._save_frame(foreground, frame_idx)

            print(
                f"[frame {frame_idx}] input={len(current_points)} "
                f"foreground={len(foreground)}"
            )

            self.stats["frames_processed"] += 1
            self.stats["total_input_points"] += len(current_points)
            self.stats["total_foreground_points"] += len(foreground)
            if visualize:
                self._visualize_foreground(foreground, frame_idx)

        result = {
            "pcap_path": str(pcap_path),
            "background_pcap_path": str(background_pcap_path),
            "frames_processed": self.stats["frames_processed"],
            "total_input_points": self.stats["total_input_points"],
            "total_foreground_points": self.stats["total_foreground_points"],
            "reference_frame": reference_frame,
            "threshold": self.threshold,
            "max_distance": max_distance,
        }

        if return_frames:
            result["foreground_frames"] = frames

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Offline Ouster PCAP background subtraction + Open3D visualization"
    )
    parser.add_argument("input_scan",type=str,help="Path to foreground/input Ouster .pcap file")
    parser.add_argument("background",type=str,help="Path to background/reference Ouster .pcap file")
    parser.add_argument("--visualize",action="store_true",help="Visualize foreground in Open3D")
    parser.add_argument("--start",type=int,default=0,help="First frame to process from the foreground PCAP")
    parser.add_argument("--frames",type=int,default=1000,help="Number of frames to process from the foreground PCAP")
    parser.add_argument("--reference-frame",type=int,default=1,help="Frame index to use from the background PCAP")
    parser.add_argument("--knn_threshold",type=float,default=0.01,help="Threshold for nearest neighbor background subtraction (in meters)")
    parser.add_argument("--max_distance", type=float, default=5, help="Maximum distance from sensor to keep points in meters (default = 5)")
    parser.add_argument("--output-dir",type=str,default=None,help="Directory to save per-frame .npy foreground files")
    parser.add_argument("--save-foreground",action="store_true",help="Whether to save foreground point clouds as .npy files in the output directory")

    args = parser.parse_args()

    processor = BinaryFrameProcessor(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        threshold=args.knn_threshold,
        visualize=args.visualize,
        save_foreground=args.save_foreground,
        max_distance=args.max_distance,
    )

    try:
        summary = processor.process_pcap(
            pcap_path=args.input_scan,
            background_pcap_path=args.background,
            max_frames=args.frames,
            start_frame=args.start,
            reference_frame=args.reference_frame,
            visualize=args.visualize,
            max_distance=args.max_distance,
        )
        print("\nDone.")
        print(summary)

        if args.visualize:
            while processor.vis is not None:
                processor.vis.poll_events()
                processor.vis.update_renderer()
    finally:
        processor.close_visualizer()


if __name__ == "__main__":
    main()