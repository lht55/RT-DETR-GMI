#!/usr/bin/env python3
"""
Regenerate manuscript tables from released per-seed metrics.

F1 aggregation used in the manuscript
--------------------------------------
For each independent run:
    1. obtain the class-averaged Precision P;
    2. obtain the class-averaged Recall R;
    3. compute manuscript F1 = 2PR / (P + R).

Then calculate the mean and sample standard deviation (n-1) across runs.

The released `class_avg_f1_pct` is retained as a distinct metric:
it is the average of per-class F1 values within each run and is not the
F1 quantity reported in the manuscript tables.

Table 6 note
------------
The GIoU and Inner-MPDIoU rows use the same complete architecture as the
corresponding Table 7 configurations:
- GIoU: GCS + MSLA-AIFI + MPMF-Net
- Inner-MPDIoU: Full model
The remaining five Table 6 loss variants are stored directly as Table 6 rows.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value == "" else float(value)


def manuscript_f1(p, r):
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


def mean_sd(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) >= 2 else 0.0


def fmt(mean, sd):
    return "" if mean is None else f"{mean:.2f} ± {sd:.2f}"


def summarize_rows(rows):
    metrics = {
        "precision_pct": [to_float(r.get("precision_pct")) for r in rows],
        "recall_pct": [to_float(r.get("recall_pct")) for r in rows],
        "manuscript_f1_pct": [
            manuscript_f1(to_float(r.get("precision_pct")), to_float(r.get("recall_pct")))
            for r in rows
        ],
        "class_avg_f1_pct": [to_float(r.get("class_avg_f1_pct")) for r in rows],
        "map50_pct": [to_float(r.get("map50_pct")) for r in rows],
        "map50_95_pct": [to_float(r.get("map50_95_pct")) for r in rows],
    }
    out = {"n_runs": len(rows)}
    for name, values in metrics.items():
        mean, sd = mean_sd(values)
        out[f"{name}_mean"] = mean
        out[f"{name}_sd"] = sd
        out[f"{name}_mean_sd"] = fmt(mean, sd)
    return out


def write_table(path, rows):
    fields = [
        "method",
        "n_runs",
        "precision_pct",
        "recall_pct",
        "f1_score_pct",
        "class_avg_f1_pct_audit",
        "map50_pct",
        "map50_95_pct",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_row(method, summary):
    return {
        "method": method,
        "n_runs": summary["n_runs"],
        "precision_pct": summary["precision_pct_mean_sd"],
        "recall_pct": summary["recall_pct_mean_sd"],
        "f1_score_pct": summary["manuscript_f1_pct_mean_sd"],
        "class_avg_f1_pct_audit": summary["class_avg_f1_pct_mean_sd"],
        "map50_pct": summary["map50_pct_mean_sd"],
        "map50_95_pct": summary["map50_95_pct_mean_sd"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reproducibility/results/per_seed_metrics.csv",
        help="Released per-seed metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="reproducibility/results/regenerated_tables",
        help="Directory for regenerated manuscript tables.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Input CSV is empty.")

    required = {"source_table", "model_or_config", "precision_pct", "recall_pct"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Audit the stored manuscript_f1_pct column if present.
    max_diff = 0.0
    if "manuscript_f1_pct" in rows[0]:
        for r in rows:
            p = to_float(r.get("precision_pct"))
            rec = to_float(r.get("recall_pct"))
            stored = to_float(r.get("manuscript_f1_pct"))
            calc = manuscript_f1(p, rec)
            if stored is not None and calc is not None:
                max_diff = max(max_diff, abs(stored - calc))

    # -------------------------
    # Regenerate manuscript Table 6
    # -------------------------
    table6_sources = [
        ("GIoU", "Table 7", "GCS + MSLA-AIFI + MPMF-Net"),
        ("Inner-CIoU", "Table 6", "inner-CIOU"),
        ("Inner-DIoU", "Table 6", "inner-DIOU"),
        ("Inner-EIoU", "Table 6", "inner-EIOU"),
        ("Inner-GIoU", "Table 6", "inner-GIOU"),
        ("MPDIoU", "Table 6", "MPDIOU"),
        ("Inner-MPDIoU", "Table 7", "Full model"),
    ]

    table6 = []
    for label, source_table, config in table6_sources:
        selected = [
            r for r in rows
            if r.get("dataset") == "PVEL-AD"
            and r.get("source_table") == source_table
            and r.get("model_or_config") == config
        ]
        if len(selected) != 3:
            raise SystemExit(
                f"Expected 3 runs for Table 6 {label}, found {len(selected)} "
                f"(source={source_table}, config={config})."
            )
        table6.append(table_row(label, summarize_rows(selected)))

    table6_path = output_dir / "table6_regenerated.csv"
    write_table(table6_path, table6)

    # -------------------------
    # Regenerate manuscript Table 11
    # -------------------------
    table11 = []
    for model in ("RT-DETR-r18", "RT-DETR-GMI"):
        selected = [
            r for r in rows
            if r.get("source_table") == "Table 11"
            and r.get("model_or_config") == model
        ]
        if len(selected) != 3:
            raise SystemExit(f"Expected 3 runs for Table 11 {model}, found {len(selected)}.")
        table11.append(table_row(model, summarize_rows(selected)))

    table11_path = output_dir / "table11_regenerated.csv"
    write_table(table11_path, table11)

    print(f"Input: {input_path}")
    print(f"Maximum stored-vs-recalculated manuscript F1 difference: {max_diff:.6f}")
    print()

    print("=== Regenerated Table 6 ===")
    for r in table6:
        print(
            f'{r["method"]}: P={r["precision_pct"]}, R={r["recall_pct"]}, '
            f'F1={r["f1_score_pct"]}, mAP50={r["map50_pct"]}, '
            f'mAP50:95={r["map50_95_pct"]} '
            f'[class-avg F1 audit: {r["class_avg_f1_pct_audit"]}]'
        )

    print()
    print("=== Regenerated Table 11 ===")
    for r in table11:
        print(
            f'{r["method"]}: P={r["precision_pct"]}, R={r["recall_pct"]}, '
            f'F1={r["f1_score_pct"]}, mAP50={r["map50_pct"]}, '
            f'mAP50:95={r["map50_95_pct"]} '
            f'[class-avg F1 audit: {r["class_avg_f1_pct_audit"]}]'
        )

    print()
    print(f"Saved: {table6_path}")
    print(f"Saved: {table11_path}")


if __name__ == "__main__":
    main()
