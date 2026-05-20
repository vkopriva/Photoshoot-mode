#!/usr/bin/env python3
"""
Detection metrics for the live pipeline.

The evaluator compares predicted 3D axis-aligned bounding boxes against
ground-truth boxes and reports 3D IoU, precision, recall, and F1 score.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Box = Dict[str, Any]


def box_3d_iou(box_a: Box, box_b: Box) -> float:
    """Compute IoU for two 3D axis-aligned bounding boxes."""
    a = normalize_box(box_a)
    b = normalize_box(box_b)

    inter_x = max(0.0, min(a["max_x"], b["max_x"]) - max(a["min_x"], b["min_x"]))
    inter_y = max(0.0, min(a["max_y"], b["max_y"]) - max(a["min_y"], b["min_y"]))
    inter_z = max(0.0, min(a["max_z"], b["max_z"]) - max(a["min_z"], b["min_z"]))
    inter_volume = inter_x * inter_y * inter_z

    if inter_volume <= 0.0:
        return 0.0

    volume_a = _box_volume(a)
    volume_b = _box_volume(b)
    union = volume_a + volume_b - inter_volume
    if union <= 0.0:
        return 0.0

    return float(inter_volume / union)


def evaluate_summary(
    summary: Dict[str, Any],
    ground_truth_path: str,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Dict[str, Any]:
    """Evaluate a MasterPipeline summary against a ground-truth annotation file."""
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = predictions_from_summary(summary, class_agnostic=class_agnostic)
    evaluated_frame_indices = {
        _metrics_frame_index(frame)
        for frame in summary.get("frame_results", [])
        if "frame_idx" in frame or "parsed_frame_idx" in frame
    }
    metrics_frame_index_source = (
        "parsed_frame_idx"
        if any("parsed_frame_idx" in frame for frame in summary.get("frame_results", []))
        else "frame_idx"
    )
    if evaluated_frame_indices:
        ground_truth = {
            frame_idx: boxes
            for frame_idx, boxes in ground_truth.items()
            if int(frame_idx) in evaluated_frame_indices
        }
    metrics = evaluate_detections(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        evaluated_frame_indices=evaluated_frame_indices,
    )
    metrics["frame_index_source"] = metrics_frame_index_source
    return metrics


def evaluate_detections(
    predictions: Sequence[Box],
    ground_truth: Dict[int, List[Box]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    evaluated_frame_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Compute aggregate and per-class detection metrics."""
    gt_by_class = _group_ground_truth(ground_truth, class_agnostic=class_agnostic)
    pred_by_class = _group_predictions(predictions, class_agnostic=class_agnostic)
    true_class_counts = {
        label: sum(len(boxes) for boxes in frames.values())
        for label, frames in gt_by_class.items()
    }

    classes = sorted(set(gt_by_class) | set(pred_by_class))
    per_class: Dict[str, Dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_gt = 0
    matched_ious: List[float] = []
    f1_values: List[float] = []

    for label in classes:
        gt_for_label = gt_by_class.get(label, {})
        preds_for_label = sorted(
            pred_by_class.get(label, []),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        matched: Dict[int, set] = {frame_idx: set() for frame_idx in gt_for_label}
        tp_flags: List[int] = []
        fp_flags: List[int] = []
        label_ious: List[float] = []
        gt_count = sum(len(boxes) for boxes in gt_for_label.values())

        for pred in preds_for_label:
            frame_idx = int(pred["frame_idx"])
            gt_boxes = gt_for_label.get(frame_idx, [])
            best_iou = 0.0
            best_gt_idx: Optional[int] = None

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched.setdefault(frame_idx, set()):
                    continue
                iou = box_3d_iou(pred, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched[frame_idx].add(best_gt_idx)
                tp_flags.append(1)
                fp_flags.append(0)
                label_ious.append(best_iou)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        precision_curve, recall_curve = _precision_recall_curve(tp_flags, fp_flags, gt_count)
        tp = int(sum(tp_flags))
        fp = int(sum(fp_flags))
        fn = int(gt_count - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mean_iou = sum(label_ious) / len(label_ious) if label_ious else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "mean_iou": mean_iou,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "num_ground_truth": gt_count,
            "num_predictions": len(preds_for_label),
        }

        if gt_count > 0:
            f1_values.append(f1)
        total_tp += tp
        total_fp += fp
        total_gt += gt_count
        matched_ious.extend(label_ious)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / total_gt if total_gt else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    average_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    evaluated_frames = sorted(int(frame_idx) for frame_idx in evaluated_frame_indices or [])

    return {
        "iou_threshold": iou_threshold,
        "class_agnostic": class_agnostic,
        "frame_index_source": "frame_idx",
        "evaluated_frames": evaluated_frames,
        "precision": overall_precision,
        "recall": overall_recall,
        "average_f1": average_f1,
        "macro_f1": average_f1,
        "mean_3d_iou": mean_iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_gt - total_tp,
        "num_ground_truth": total_gt,
        "num_predictions": len(predictions),
        "true_class_counts": true_class_counts,
        "per_class": per_class,
    }


def predictions_from_summary(
    summary: Dict[str, Any],
    class_agnostic: bool = False,
) -> List[Box]:
    """Extract predicted boxes and class scores from a MasterPipeline summary."""
    predictions: List[Box] = []

    for frame in summary.get("frame_results", []):
        frame_idx = _metrics_frame_index(frame)
        boxes = frame.get("bounding_boxes", [])
        classifications = frame.get("classifications", [])

        for idx, bbox in enumerate(boxes):
            label = "object"
            score = 1.0
            if idx < len(classifications):
                label, score = classifications[idx]

            if label == "Not a class" and not class_agnostic:
                continue

            pred = normalize_box(bbox)
            pred["frame_idx"] = frame_idx
            if "original_frame_idx" in frame:
                pred["original_frame_idx"] = int(frame["original_frame_idx"])
            elif "frame_idx" in frame:
                pred["original_frame_idx"] = int(frame["frame_idx"])
            if "parsed_frame_idx" in frame:
                pred["parsed_frame_idx"] = int(frame["parsed_frame_idx"])
            pred["label"] = "object" if class_agnostic else str(label)
            pred["score"] = float(score)
            predictions.append(pred)

    return predictions


def _metrics_frame_index(frame: Dict[str, Any]) -> int:
    """Return the frame index used for metric matching against ground truth."""
    if "parsed_frame_idx" in frame:
        return int(frame["parsed_frame_idx"])
    return int(frame["frame_idx"])


def load_ground_truth(path: str) -> Dict[int, List[Box]]:
    """Load ground-truth boxes from JSON or CSV."""
    gt_path = Path(path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    if gt_path.suffix.lower() == ".csv":
        return _load_ground_truth_csv(gt_path)

    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _load_ground_truth_json(data)


def normalize_box(box: Box) -> Box:
    """Return a box with min/max coordinates, preserving label and score fields."""
    result = dict(box)

    if all(key in result for key in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")):
        result["min_x"] = float(result["min_x"])
        result["max_x"] = float(result["max_x"])
        result["min_y"] = float(result["min_y"])
        result["max_y"] = float(result["max_y"])
        result["min_z"] = float(result["min_z"])
        result["max_z"] = float(result["max_z"])
        return result

    if isinstance(result.get("center"), dict):
        center = result["center"]
        result["center_x"] = center.get("x", center.get("center_x"))
        result["center_y"] = center.get("y", center.get("center_y"))
        result["center_z"] = center.get("z", center.get("center_z"))

    center_keys = ("center_x", "center_y", "center_z")
    size_keys = ("width", "depth", "height")
    length_size_keys = ("width", "length", "height")
    alt_size_keys = ("size_x", "size_y", "size_z")
    dim_keys = ("dx", "dy", "dz")

    # Helper to safely convert to float, returning None if value is None or missing
    def safe_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    sx, sy, sz = None, None, None
    if all(key in result for key in center_keys) and all(key in result for key in size_keys):
        sx, sy, sz = safe_float(result.get("width")), safe_float(result.get("depth")), safe_float(result.get("height"))
    elif all(key in result for key in center_keys) and all(key in result for key in length_size_keys):
        sx, sy, sz = safe_float(result.get("width")), safe_float(result.get("length")), safe_float(result.get("height"))
    elif all(key in result for key in center_keys) and all(key in result for key in alt_size_keys):
        sx, sy, sz = safe_float(result.get("size_x")), safe_float(result.get("size_y")), safe_float(result.get("size_z"))
    elif all(key in result for key in center_keys) and all(key in result for key in dim_keys):
        sx, sy, sz = safe_float(result.get("dx")), safe_float(result.get("dy")), safe_float(result.get("dz"))

    # If any dimension is None, return None to signal this box should be skipped
    if sx is None or sy is None or sz is None:
        return None

    cx = float(result["center_x"])
    cy = float(result["center_y"])
    cz = float(result["center_z"])
    result["min_x"] = cx - sx / 2.0
    result["max_x"] = cx + sx / 2.0
    result["min_y"] = cy - sy / 2.0
    result["max_y"] = cy + sy / 2.0
    result["min_z"] = cz - sz / 2.0
    result["max_z"] = cz + sz / 2.0
    return result


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print a concise metrics report."""
    print("\n==========================================")
    print(" DETECTION METRICS")
    print("==========================================")
    print(f"IoU threshold    : {metrics['iou_threshold']:.3f}")
    print(f"Class agnostic   : {metrics['class_agnostic']}")
    print(f"Frame index      : {metrics.get('frame_index_source', 'frame_idx')}")
    if metrics.get("evaluated_frames"):
        frames = metrics["evaluated_frames"]
        print(f"Evaluated frames : {len(frames)} ({frames[0]}..{frames[-1]})")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"Average F1       : {metrics.get('average_f1', metrics.get('macro_f1', 0.0)):.4f}")
    print(f"Mean 3D IoU      : {metrics['mean_3d_iou']:.4f}")
    print(f"TP / FP / FN     : {metrics['true_positives']} / {metrics['false_positives']} / {metrics['false_negatives']}")

    # Build comparison table: true class counts vs classifications
    true_class_counts = metrics.get("true_class_counts", {})
    per_class = metrics.get("per_class", {})

    # Get all classes: those in ground truth + those with predictions but no ground truth
    gt_classes = set(true_class_counts.keys())
    pred_classes = set(per_class.keys())
    all_classes = gt_classes | pred_classes

    # Separate into two groups:
    # - matched: classes that exist in ground truth
    # - extra: classes that have predictions but no ground truth
    matched_classes = sorted(gt_classes)
    extra_classes = sorted(pred_classes - gt_classes)

    if all_classes:
        # Print comparison table
        print("\nClass Comparison:")
        print("-" * 74)
        print(
            f"{'Class':<20} {'True Count':>12} {'Classifications':>15} "
            f"{'TP':>6} {'FP':>6} {'FN':>6}"
        )
        print("-" * 74)

        # Rows for classes that exist in ground truth
        for label in matched_classes:
            true_count = true_class_counts.get(label, 0)
            class_metrics = per_class.get(label, {})
            pred_count = class_metrics.get("num_predictions", 0)
            tp = class_metrics.get("true_positives", 0)
            fp = class_metrics.get("false_positives", 0)
            fn = class_metrics.get("false_negatives", 0)
            print(f"{label:<20} {true_count:>12} {pred_count:>15} {tp:>6} {fp:>6} {fn:>6}")

        # Rows for classes with predictions but no ground truth
        for label in extra_classes:
            class_metrics = per_class.get(label, {})
            pred_count = class_metrics.get("num_predictions", 0)
            tp = class_metrics.get("true_positives", 0)
            fp = class_metrics.get("false_positives", 0)
            fn = class_metrics.get("false_negatives", 0)
            print(f"{label:<20} {'-':>12} {pred_count:>15} {tp:>6} {fp:>6} {fn:>6}")

        print("-" * 74)
    else:
        print("\nClass Comparison: None")

    print("\nPer class:")
    if metrics["per_class"]:
        for label, values in sorted(metrics["per_class"].items()):
            print(
                f"  {label}: P={values['precision']:.4f}, R={values['recall']:.4f}, "
                f"F1={values.get('f1', 0.0):.4f}, "
                f"IoU={values['mean_iou']:.4f}"
            )
    else:
        print("  None")
    print("==========================================")


def plot_precision_recall_curves(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot per-class precision-recall curves stored in a metrics dict."""
    import matplotlib.pyplot as plt

    per_class = metrics.get("per_class", {})
    curve_items = []
    for label, values in sorted(per_class.items()):
        if values.get("num_ground_truth", 0) <= 0:
            continue

        precisions = values.get("precision_curve", [])
        recalls = values.get("recall_curve", [])
        if not precisions or not recalls:
            continue

        plot_recalls = [0.0, *[float(v) for v in recalls]]
        plot_precisions = [1.0, *[float(v) for v in precisions]]
        curve_items.append((label, plot_recalls, plot_precisions, values))

    if not curve_items:
        print("[metrics] No precision-recall curve data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, recalls, precisions, values in curve_items:
        ax.plot(
            recalls,
            precisions,
            drawstyle="steps-post",
            linewidth=2,
            label=label,
        )

    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] Precision-recall curve saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _load_ground_truth_csv(path: Path) -> Dict[int, List[Box]]:
    frames: Dict[int, List[Box]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row.get("frame_idx", row.get("frame", 0)))
            label = row.get("label", "object")
            box = normalize_box({key: value for key, value in row.items() if value not in (None, "")})
            box["label"] = label
            frames.setdefault(frame_idx, []).append(box)
    return frames


def _load_ground_truth_json(data: Any) -> Dict[int, List[Box]]:
    frames: Dict[int, List[Box]] = {}

    if isinstance(data, dict) and "frames" in data:
        iterable = data["frames"]
    elif isinstance(data, dict):
        iterable = [{"frame_idx": key, "boxes": value} for key, value in data.items()]
    elif isinstance(data, list):
        iterable = data
    else:
        raise ValueError("Ground-truth JSON must be a list, a frame map, or contain a 'frames' list.")

    for frame_item in iterable:
        if not isinstance(frame_item, dict):
            raise ValueError(f"Invalid frame annotation: {frame_item}")

        frame_idx = _frame_index_from_item(frame_item)
        boxes = frame_item.get(
            "boxes",
            frame_item.get(
                "bounding_boxes",
                frame_item.get("objects", frame_item.get("annotations", [])),
            ),
        )
        if isinstance(boxes, dict):
            boxes = [boxes]

        for item in boxes:
            label = item.get("label", item.get("class", item.get("object_id", "object")))
            raw_box = item.get("bbox", item.get("box", item))
            box = normalize_box(raw_box)
            if box is None:
                continue  # Skip boxes with missing/invalid dimensions
            box["label"] = str(label)
            frames.setdefault(frame_idx, []).append(box)

    return frames


def _group_ground_truth(
    ground_truth: Dict[int, List[Box]],
    class_agnostic: bool,
) -> Dict[str, Dict[int, List[Box]]]:
    grouped: Dict[str, Dict[int, List[Box]]] = {}
    for frame_idx, boxes in ground_truth.items():
        for box in boxes:
            label = "object" if class_agnostic else str(box.get("label", "object"))
            normalized = normalize_box(box)
            if normalized is None:
                continue  # Skip boxes with missing/invalid dimensions
            grouped.setdefault(label, {}).setdefault(int(frame_idx), []).append(normalized)
    return grouped


def _frame_index_from_item(frame_item: Dict[str, Any]) -> int:
    if "frame_idx" in frame_item:
        return int(frame_item["frame_idx"])
    if "frame" in frame_item:
        return int(frame_item["frame"])
    if "filename" in frame_item:
        return int(Path(str(frame_item["filename"])).stem)
    return 0


def _group_predictions(
    predictions: Sequence[Box],
    class_agnostic: bool,
) -> Dict[str, List[Box]]:
    grouped: Dict[str, List[Box]] = {}
    for pred in predictions:
        label = "object" if class_agnostic else str(pred.get("label", "object"))
        grouped.setdefault(label, []).append(normalize_box(pred))
    return grouped


def _precision_recall_curve(
    tp_flags: Sequence[int],
    fp_flags: Sequence[int],
    gt_count: int,
) -> Tuple[List[float], List[float]]:
    precisions: List[float] = []
    recalls: List[float] = []
    tp_total = 0
    fp_total = 0

    for tp, fp in zip(tp_flags, fp_flags):
        tp_total += tp
        fp_total += fp
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        recall = tp_total / gt_count if gt_count else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def _legacy_unused_curve_area(precisions: Sequence[float], recalls: Sequence[float]) -> float:
    if not precisions:
        return 0.0

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]

    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    area = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            area += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return float(area)


def _box_volume(box: Box) -> float:
    return max(0.0, box["max_x"] - box["min_x"]) * max(0.0, box["max_y"] - box["min_y"]) * max(0.0, box["max_z"] - box["min_z"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3D detection metrics from saved predictions and ground truth.")
    parser.add_argument("--predictions", type=str, required=True, help="JSON file containing a MasterPipeline summary")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground-truth JSON or CSV annotation file")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Minimum 3D IoU for a true positive")
    parser.add_argument("--class-agnostic", action="store_true", help="Ignore class labels and evaluate boxes only")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics JSON")
    parser.add_argument("--no-pr-curve", action="store_true", help="Do not show the precision-recall curve after metrics are computed")
    parser.add_argument("--pr-curve-output", type=str, default=None, help="Optional path to save the precision-recall curve image")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = evaluate_summary(
        summary=summary,
        ground_truth_path=args.ground_truth,
        iou_threshold=args.iou_threshold,
        class_agnostic=args.class_agnostic,
    )
    print_metrics(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    plot_precision_recall_curves(
        metrics,
        save_path=args.pr_curve_output,
        show=not args.no_pr_curve,
    )


if __name__ == "__main__":
    main()
