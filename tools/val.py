"""Evaluate a local RT-DETR checkpoint and export reproducible run-level results.

Run from the repository root:
    python tools/val.py --model runs/train/GMI_seed0/weights/best.pt --data path/to/data.yaml
    python tools/val.py --model path/to/best.pt --data path/to/data.yaml --dry-run

--cfg accepts evaluation arguments in YAML; explicit CLI arguments take priority.
P/R follow the repository's maximum-smoothed-mean-F1 operating point on the
evaluated split. --conf does not turn these into fixed-threshold P/R metrics.
Matrix confidence/IoU thresholds are separate (--cm-conf / --cm-iou).
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULTS = {
    "split": "test", "imgsz": 640, "batch": 16, "workers": 4,
    "conf": 0.001, "iou": 0.7, "max_det": 300,
    "half": False, "plots": True, "rect": True, "agnostic_nms": False,
    "save_json": False, "save_txt": False, "save_conf": False,
    "project": "runs/val", "name": "exp", "exist_ok": False,
    "cm_conf": 0.25, "cm_iou": 0.45, "profile": True,
}


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cfg", help="Optional evaluation-argument YAML.")
    parser.add_argument("--model", "--weights", dest="model", help="Trained local .pt checkpoint.")
    parser.add_argument("--data", help="Local dataset YAML with the requested split.")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--device", help="GPU such as 0, or cpu.")
    parser.add_argument("--project", help="Output project directory (default: runs/val).")
    parser.add_argument("--name", help="Run name, for example GMI_seed0_test.")
    for key in ("imgsz", "batch", "workers", "max-det"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--conf", type=float, help="Framework argument; RT-DETR retains decoder queries for AP.")
    parser.add_argument("--iou", type=float, help="Framework NMS argument; this RT-DETR postprocessor does not use NMS.")
    parser.add_argument("--cm-conf", type=float, help="Matrix confidence threshold (default: > 0.25).")
    parser.add_argument("--cm-iou", type=float, help="Matrix matching IoU threshold (default: > 0.45).")
    for key in ("half", "plots", "profile", "save-json", "save-txt", "save-conf", "exist-ok"):
        parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true", help="Check file paths and print settings without inference.")
    return parser


def local_file(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Provide {label} as a nonempty local file path")
    path = Path(value).expanduser()
    for candidate in (path, REPO_ROOT / path):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"{label} not found: {value}")


def resolve_settings(args):
    cli = vars(args).copy()
    dry_run = cli.pop("dry_run", False)
    config_path = cli.pop("cfg", None)
    config = {}
    if config_path:
        with open(local_file(config_path, "Evaluation configuration"), encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("Evaluation configuration must be a YAML mapping")
        if "backbone" in config or "head" in config:
            raise ValueError("Evaluation requires a trained .pt checkpoint, not an architecture YAML")
        config.pop("save_dir", None)
        config.pop("cfg", None)
    settings = {**DEFAULTS, **config, **cli, "task": "detect", "mode": "val"}
    settings["model"] = local_file(settings.get("model"), "--model checkpoint")
    settings["data"] = local_file(settings.get("data"), "--data dataset YAML")
    if Path(settings["model"]).suffix.lower() != ".pt":
        raise ValueError("--model must be a trained .pt checkpoint; model YAMLs are for train.py")
    if Path(settings["data"]).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("--data must be a dataset .yaml or .yml file")
    with open(REPO_ROOT / "ultralytics/cfg/default.yaml", encoding="utf-8") as stream:
        known_keys = set(yaml.safe_load(stream)) | {"cm_conf", "cm_iou"}
    unknown = set(settings) - known_keys
    if unknown:
        raise ValueError(f"Unknown evaluation arguments: {', '.join(sorted(map(str, unknown)))}")
    for key in ("conf", "iou", "cm_conf", "cm_iou"):
        if not isinstance(settings[key], (int, float)) or not 0 <= settings[key] <= 1:
            raise ValueError(f"{key} must be a number in [0, 1]")
    for key in ("imgsz", "batch", "max_det"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["split"] not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    if not isinstance(settings["profile"], bool):
        raise ValueError("profile must be a YAML boolean (true or false)")
    return settings, dry_run


def write_csv(path, records):
    if not records:
        return
    # A BOM allows Excel on Windows to display non-ASCII class names correctly.
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_records(result, validator):
    """Use actual class IDs; missing class metrics remain blank rather than invented."""
    import numpy as np

    box = result.box
    names = {int(key): str(value) for key, value in validator.names.items()}
    support = np.asarray(validator.nt_per_class, dtype=np.int64)
    indices = {int(class_id): index for index, class_id in enumerate(box.ap_class_index)}
    rows = []
    for class_id, name in sorted(names.items()):
        index = indices.get(class_id)
        row = {"class_id": class_id, "class_name": name, "support": int(support[class_id]),
               "metrics_available": index is not None}
        values = (box.p[index], box.r[index], box.f1[index], box.ap50[index],
                  box.all_ap[index, 5], box.ap[index]) if index is not None else (None,) * 6
        for key, value in zip(("precision_pct", "recall_pct", "f1_score_pct", "map50_pct", "map75_pct", "map50_95_pct"), values):
            row[key] = float(value) * 100 if value is not None else None
        rows.append(row)
    precision, recall = float(box.mp), float(box.mr)
    summary = {
        "images": int(validator.seen), "support": int(support.sum()),
        "precision_pct": precision * 100, "recall_pct": recall * 100,
        "class_avg_f1_pct": float(np.mean(box.f1)) * 100 if len(box.f1) else None,
        "manuscript_f1_pct": 200 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50_pct": float(box.map50) * 100, "map75_pct": float(box.map75) * 100,
        "map50_95_pct": float(box.map) * 100,
    }
    return rows, summary


def selected_confidence(result):
    """Reconstruct the repository's curve index and verify it matches the returned P/R."""
    import numpy as np
    from ultralytics.utils.metrics import smooth

    box = result.box
    curve = np.asarray(getattr(box, "f1_curve", []))
    grid = np.asarray(getattr(box, "px", []))
    if curve.ndim != 2 or not curve.shape[0] or grid.shape != (curve.shape[1],):
        return None
    index = int(smooth(curve.mean(0), 0.1).argmax())
    if not (np.allclose(np.asarray(box.p_curve)[:, index], box.p)
            and np.allclose(np.asarray(box.r_curve)[:, index], box.r)):
        raise RuntimeError("Returned P/R do not match the reconstructed maximum-F1 curve index")
    return float(grid[index])


def export_results(model, validator, settings):
    import numpy as np
    import torch
    import ultralytics
    from prettytable import PrettyTable
    from ultralytics.utils.torch_utils import get_flops

    result = validator.metrics
    save_dir = Path(validator.save_dir)
    rows, summary = metric_records(result, validator)
    cm = validator.confusion_matrix
    matrix = np.asarray(cm.matrix)
    nc = len(validator.names)
    if matrix.shape != (nc + 1, nc + 1) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Unexpected confusion-matrix shape or nonfinite counts")
    if np.any(matrix < 0) or not np.allclose(matrix, np.rint(matrix)):
        raise RuntimeError("Confusion-matrix counts must be nonnegative integers")
    counts = np.rint(matrix).astype(np.int64)
    support_rows = [{"class_id": row["class_id"], "class_name": row["class_name"],
                     "support": row["support"], "cm_true_column_total": int(counts[:, row["class_id"]].sum())}
                    for row in rows]
    if any(row["support"] != row["cm_true_column_total"] for row in support_rows):
        raise RuntimeError("Confusion-matrix true-column totals do not match ground-truth supports")
    labels = [row["class_name"] for row in rows] + ["background"]
    normalized = counts / (counts.sum(0, keepdims=True) + 1e-9)
    for filename, array in (("confusion_matrix_counts.csv", counts), ("confusion_matrix_normalized.csv", normalized)):
        with open(save_dir / filename, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["predicted_class / true_class", *labels])
            for name, values in zip(labels, array.tolist()):
                writer.writerow([name, *values])

    speed = {key: float(value) for key, value in validator.speed.items()}
    total_ms = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))
    checkpoint_args = (model.ckpt or {}).get("train_args") or getattr(model.model, "args", {}) or {}
    training_seed = checkpoint_args.get("seed")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    gflops = float(get_flops(model.model, validator.args.imgsz)) if settings["profile"] else None
    if gflops is not None and (not np.isfinite(gflops) or gflops <= 0):
        gflops = None
    threshold = selected_confidence(result)
    summary = {
        "run_name": save_dir.name, "checkpoint": settings["model"], "checkpoint_training_seed": training_seed,
        "split": validator.args.split, **summary,
        "max_f1_confidence": threshold, "parameters": parameters, "gflops": gflops,
        "preprocess_ms_per_image": speed.get("preprocess"), "inference_ms_per_image": speed.get("inference"),
        "postprocess_ms_per_image": speed.get("postprocess"),
        "fps_end_to_end": 1000 / total_ms if total_ms > 0 else None,
        "fps_inference_only": 1000 / speed["inference"] if speed.get("inference", 0) > 0 else None,
        "weight_size_mib": Path(settings["model"]).stat().st_size / 1024**2,
    }
    for row in rows:
        row.update(checkpoint_training_seed=training_seed, split=validator.args.split)
    write_csv(save_dir / "summary_metrics.csv", [summary])
    write_csv(save_dir / "per_class_metrics.csv", rows)
    write_csv(save_dir / "class_support.csv", support_rows)

    actual_args = vars(validator.args).copy()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
        "checkpoint": {"path": settings["model"], "sha256": file_hash(settings["model"]), "training_seed": training_seed},
        "dataset_yaml": {"path": settings["data"], "sha256": file_hash(settings["data"])},
        "requested_settings": settings, "actual_validation_settings": actual_args,
        "device": str(validator.device), "class_names": validator.names,
        "dataset_split_paths": {key: validator.data.get(key) for key in ("train", "val", "test")},
        "map_matching_iou_thresholds": [round(float(value), 2) for value in validator.iouv.cpu().tolist()],
        "operating_point": {
            "selection_split": validator.args.split,
            "protocol": "Maximum smoothed mean per-class F1 on the evaluated split; smoothing fraction 0.1.",
            "confidence": threshold,
            "precision_recall_matching_iou": 0.5,
            "class_aggregation": "Unweighted mean over class IDs returned by ap_per_class.",
            "class_avg_f1_pct": "Mean of per-class F1 at the selected common confidence, multiplied by 100.",
            "manuscript_f1_pct": "2 * class-mean precision * class-mean recall / (precision + recall), multiplied by 100.",
        },
        "confusion_matrix": {
            "confidence_threshold": float(cm.conf), "matching_iou_threshold": float(cm.iou_thres),
            "confidence_comparison": ">", "iou_comparison": ">",
            "rows": "predicted classes", "columns": "ground-truth classes",
            "background_position": "last row and last column",
            "normalization": "Each true-class column divided by its sum plus 1e-9; includes background column.",
            "counts_origin": "Direct export from this run's repository ConfusionMatrix.matrix.",
            "legacy_background_behavior": "The archived validator skips matrix updates for label-free images; "
                "the archived matrix records unmatched detections only when at least one IoU match exists. "
                "Background false-positive counts therefore follow this legacy implementation.",
        },
        "rtdetr_postprocessing": "This repository keeps the RT-DETR decoder queries for AP; conf/iou/max_det "
            "framework arguments do not apply YOLO-style confidence filtering or NMS here. Matrix thresholds apply separately.",
        "metric_units": "Detection metric CSV columns ending in _pct are percentages; matrix normalization is in [0,1].",
        "missing_metrics": "Blank class metrics mean the repository did not return metrics for that class; "
            "this includes absent classes and runs for which its metrics-processing gate was not entered.",
        "model_statistics": {"parameters": parameters, "gflops": gflops,
                             "parameter_state": "Model used by AutoBackend after its setup/fusion."},
        "timing": {"speed_ms_per_image": speed, "fps_end_to_end": summary["fps_end_to_end"],
                   "fps_definition": "1000 / (preprocess + inference + postprocess milliseconds per image), measured on this run."},
    }
    with open(save_dir / "evaluation_settings.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    write_csv(save_dir / "evaluation_settings.csv", [{
        "checkpoint": settings["model"], "checkpoint_training_seed": training_seed, "data": settings["data"],
        "split": validator.args.split, "device": str(validator.device), "imgsz": validator.args.imgsz,
        "batch": validator.args.batch, "workers": validator.args.workers, "half": validator.args.half,
        "conf_argument": validator.args.conf, "iou_argument": validator.args.iou, "max_det_argument": validator.args.max_det,
        "plots": validator.args.plots, "max_f1_confidence": threshold,
        "cm_conf": float(cm.conf), "cm_iou": float(cm.iou_thres),
        "cm_conf_comparison": ">", "cm_iou_comparison": ">",
        "operating_point_protocol": "Maximum smoothed mean class F1 on evaluated split",
        "matrix_axes": "rows=predicted; columns=true; background=last",
    }])

    table = PrettyTable(["Class", "Support", "P (%)", "R (%)", "Class F1 (%)", "AP50 (%)", "AP75 (%)", "AP50:95 (%)"])
    def display(value):
        return "N/A" if value is None else f"{value:.4f}"
    for row in rows:
        table.add_row([row["class_name"], row["support"], *[display(row[key]) for key in
            ("precision_pct", "recall_pct", "f1_score_pct", "map50_pct", "map75_pct", "map50_95_pct")]])
    table.add_row(["all (class mean)", summary["support"], *[display(summary[key]) for key in
        ("precision_pct", "recall_pct", "class_avg_f1_pct", "map50_pct", "map75_pct", "map50_95_pct")]])
    info = PrettyTable(["Item", "Value"])
    for key in ("parameters", "gflops", "weight_size_mib", "fps_end_to_end", "fps_inference_only"):
        info.add_row([key, summary[key] if summary[key] is not None else "N/A"])
    text = (f"{info}\n\n{table}\n\nclass_avg_f1_pct: {display(summary['class_avg_f1_pct'])}\n"
            f"manuscript_f1_pct: {display(summary['manuscript_f1_pct'])}\n"
            f"P/R operating point: evaluated-split maximum smoothed mean class F1; confidence={threshold}\n"
            f"Confusion matrix: confidence > {cm.conf}, matching IoU > {cm.iou_thres}\n"
            "Matrix rows are predicted classes; columns are true classes. Detection metrics above are percentages.\n")
    (save_dir / "paper_data.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Results and numerical exports saved to: {save_dir.resolve()}")


def main(argv=None):
    parser = make_parser()
    try:
        settings, dry_run = resolve_settings(parser.parse_args(argv))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print("Resolved evaluation settings (CLI > --cfg > script defaults):", flush=True)
    print(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), flush=True)
    if dry_run:
        print("Dry run complete: checkpoint/data paths checked; no weights or dataset images were loaded.")
        return 0

    from ultralytics import RTDETR
    from ultralytics.cfg import get_cfg
    from ultralytics.models.rtdetr.val import RTDETRValidator

    arguments = {key: value for key, value in settings.items() if key not in ("cm_conf", "cm_iou", "profile")}
    get_cfg(overrides=arguments.copy())
    class ExportValidator(RTDETRValidator):
        def init_metrics(self, backend):
            super().init_metrics(backend)
            if self.nc != self.data["nc"]:
                raise ValueError(f"Checkpoint has {self.nc} classes, but dataset YAML has {self.data['nc']}")
            self.confusion_matrix.conf = settings["cm_conf"]
            self.confusion_matrix.iou_thres = settings["cm_iou"]

        def update_metrics(self, predictions, batch):
            # The archived implementation gates matrix accumulation on plots.
            # Retain its exact matching logic while also exporting counts when
            # PNG/curve generation has been disabled using --no-plots.
            draw_plots = self.args.plots
            self.args.plots = True
            try:
                super().update_metrics(predictions, batch)
            finally:
                self.args.plots = draw_plots

    model = RTDETR(settings["model"])
    validator = ExportValidator(args=arguments)
    validator(model=model.model)
    export_results(model, validator, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
