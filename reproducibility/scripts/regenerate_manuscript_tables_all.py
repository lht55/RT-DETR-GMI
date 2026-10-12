#!/usr/bin/env python3
"""
Regenerate manuscript Tables 3–11 from the released machine-readable results.

Inputs
------
reproducibility/results/per_seed_metrics.csv
reproducibility/results/per_class_metrics.csv

Outputs
-------
table3_regenerated.csv
table4_regenerated.csv
table5_regenerated.csv
table6_regenerated.csv
table7_regenerated.csv
table8_regenerated.csv
table9_regenerated.csv
table10_regenerated.csv
table11_regenerated.csv
f1_aggregation_audit.csv

F1 convention used in the manuscript
------------------------------------
For each independent run:
    1. Precision is averaged across classes.
    2. Recall is averaged across classes.
    3. Manuscript F1 is computed as 2PR / (P + R).

The reported manuscript Mean ± SD is then calculated across the three
independent run-level F1 values.

The released `class_avg_f1_pct` is a distinct metric: it is the mean of
the per-class F1 values within a run. It is retained only for auditing and
must not be substituted for the manuscript F1.

Standard deviation
------------------
Sample standard deviation (n - 1) is used across independent runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SEED_ORDER = ("0", "42", "3407")

TABLE4_ORDER = (
    "1 branch; kernels=(3)",
    "2 branches; kernels=(3,3)",
    "4 branches; kernels=(3,3,3,3)",
)

TABLE5_ORDER = (
    "4 branches; kernels=(3, 3, 3, 3)",
    "4 branches; kernels=(1, 3, 5, 7)",
    "4 branches; kernels=(3, 5, 7, 9)",
    "4 branches; kernels=(5, 7, 9, 11)",
)

TABLE7_ORDER = (
    "Baseline",
    "GCS Backbone",
    "MSLA-AIFI",
    "MPMF-Net",
    "Inner-MPDIoU",
    "GCS + MSLA-AIFI",
    "GCS + MSLA-AIFI + MPMF-Net",
    "Full model",
)

TABLE8_ORDER = (
    "RT-DETR-r18",
    "RT-DETR-r34",
    "RT-DETR-r50",
    "YOLOv8n",
    "YOLOv8s",
    "YOLO10n",
    "YOLO11n",
    "YOLO12n",
    "YOLO13n",
    "D-FINE-N",
    "D-FINE-S",
    "RT-DETR-GMI",
)

# Deterministic parameter counts documented for the MSLA architecture-selection
# variants. These are not repeated-run quantities.
TABLE4_PARAMS_M = {
    "1 branch; kernels=(3)": "13.86",
    "2 branches; kernels=(3,3)": "13.67",
    "4 branches; kernels=(3,3,3,3)": "13.62",
}

TABLE5_PARAMS_M = {
    "4 branches; kernels=(3, 3, 3, 3)": "13.62",
    "4 branches; kernels=(1, 3, 5, 7)": "13.62",
    "4 branches; kernels=(3, 5, 7, 9)": "13.62",
    "4 branches; kernels=(5, 7, 9, 11)": "13.63",
}

TABLE6_SOURCES = (
    ("GIoU", "Table 7", "GCS + MSLA-AIFI + MPMF-Net"),
    ("Inner-GIoU", "Table 6", "inner-GIOU"),
    ("Inner-DIoU", "Table 6", "inner-DIOU"),
    ("Inner-CIoU", "Table 6", "inner-CIOU"),
    ("Inner-EIoU", "Table 6", "inner-EIOU"),
    ("MPDIoU", "Table 6", "MPDIOU"),
    ("Inner-MPDIoU", "Table 7", "Full model"),
)

TABLE10_STEPS = (
    ("Baseline -> +GCS", "Baseline", "GCS Backbone"),
    ("+GCS -> +MSLA-AIFI", "GCS Backbone", "GCS + MSLA-AIFI"),
    (
        "+MSLA-AIFI -> +MPMF-Net",
        "GCS + MSLA-AIFI",
        "GCS + MSLA-AIFI + MPMF-Net",
    ),
    (
        "+MPMF-Net -> +Inner-MPDIoU",
        "GCS + MSLA-AIFI + MPMF-Net",
        "Full model",
    ),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def manuscript_f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


def mean_sd(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return mean, sd


def fmt_mean_sd(mean: Optional[float], sd: Optional[float]) -> str:
    if mean is None:
        return ""
    return f"{mean:.2f} ± {sd:.2f}"


def fmt_single(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def first_nonempty(rows: Sequence[Dict[str, str]], field: str) -> str:
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def mean_of_field(rows: Sequence[Dict[str, str]], field: str) -> Optional[float]:
    vals = [to_float(r.get(field)) for r in rows]
    mean, _ = mean_sd(vals)
    return mean


def select_runs(
    rows: Sequence[Dict[str, str]],
    source_table: str,
    model_or_config: str,
    dataset: Optional[str] = None,
) -> List[Dict[str, str]]:
    selected = [
        r
        for r in rows
        if r.get("source_table") == source_table
        and r.get("model_or_config") == model_or_config
        and (dataset is None or r.get("dataset") == dataset)
    ]
    seed_rank = {seed: i for i, seed in enumerate(SEED_ORDER)}
    selected.sort(key=lambda r: seed_rank.get(str(r.get("seed_or_run", "")), 999))
    return selected


def require_three(rows: Sequence[Dict[str, str]], label: str) -> None:
    if len(rows) != 3:
        raise ValueError(f"{label}: expected 3 independent runs, found {len(rows)}.")


def summarize_runs(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    require_three(rows, rows[0].get("model_or_config", "configuration") if rows else "configuration")

    p = [to_float(r.get("precision_pct")) for r in rows]
    r = [to_float(x.get("recall_pct")) for x in rows]
    f1 = [manuscript_f1(pp, rr) for pp, rr in zip(p, r)]
    class_f1 = [to_float(x.get("class_avg_f1_pct")) for x in rows]
    map50 = [to_float(x.get("map50_pct")) for x in rows]
    map5095 = [to_float(x.get("map50_95_pct")) for x in rows]
    fps = [to_float(x.get("end_to_end_fps")) for x in rows]

    p_m, p_s = mean_sd(p)
    r_m, r_s = mean_sd(r)
    f_m, f_s = mean_sd(f1)
    cf_m, cf_s = mean_sd(class_f1)
    m50_m, m50_s = mean_sd(map50)
    m95_m, m95_s = mean_sd(map5095)
    fps_m, fps_s = mean_sd(fps)

    return {
        "precision": fmt_mean_sd(p_m, p_s),
        "recall": fmt_mean_sd(r_m, r_s),
        "f1_manuscript": fmt_mean_sd(f_m, f_s),
        "f1_class_avg_audit": fmt_mean_sd(cf_m, cf_s) if cf_m is not None else "",
        "map50": fmt_mean_sd(m50_m, m50_s),
        "map50_95": fmt_mean_sd(m95_m, m95_s),
        "params_m": first_nonempty(rows, "params_m"),
        "gflops_g": first_nonempty(rows, "gflops_g"),
        # Manuscript tables use integer FPS.
        "fps": "" if fps_m is None else str(int(round(fps_m))),
        "fps_mean_raw": "" if fps_m is None else f"{fps_m:.6f}",
        "fps_sd_raw": "" if fps_s is None else f"{fps_s:.6f}",
    }


def standard_table_row(label: str, summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "method": label,
        "Precision (%)": summary["precision"],
        "Recall (%)": summary["recall"],
        "F1 (%)": summary["f1_manuscript"],
        "mAP50 (%)": summary["map50"],
        "mAP50:95 (%)": summary["map50_95"],
        "Params (M)": summary["params_m"],
        "GFLOPs (G)": summary["gflops_g"],
        "FPS": summary["fps"],
    }


def build_table3(seed_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    baseline = select_runs(seed_rows, "Table 8", "RT-DETR-r18", "PVEL-AD")
    gmi = select_runs(seed_rows, "Table 8", "RT-DETR-GMI", "PVEL-AD")
    require_three(baseline, "Table 3 RT-DETR-r18")
    require_three(gmi, "Table 3 RT-DETR-GMI")

    by_seed_base = {str(r["seed_or_run"]): r for r in baseline}
    by_seed_gmi = {str(r["seed_or_run"]): r for r in gmi}

    out = []
    for metric_name, field in (("mAP50", "map50_pct"), ("mAP50:95", "map50_95_pct")):
        diffs = []
        seed_diffs = {}
        for seed in SEED_ORDER:
            diff = float(by_seed_gmi[seed][field]) - float(by_seed_base[seed][field])
            diffs.append(diff)
            seed_diffs[seed] = diff

        mean_diff = statistics.mean(diffs)
        sd_diff = statistics.stdev(diffs)
        n = len(diffs)
        df = n - 1
        se = sd_diff / math.sqrt(n)
        t_stat = mean_diff / se

        if df != 2:
            raise ValueError("Table 3 reproduction currently expects exactly three paired runs (df=2).")

        # Exact two-sided Student-t p-value for df=2:
        # p = 1 - |t| / sqrt(t^2 + 2)
        p_value = 1.0 - abs(t_stat) / math.sqrt(t_stat * t_stat + 2.0)

        # 97.5th percentile of t distribution with df=2.
        t_critical_975_df2 = 4.302652729911275
        ci_low = mean_diff - t_critical_975_df2 * se
        ci_high = mean_diff + t_critical_975_df2 * se
        cohen_dz = mean_diff / sd_diff

        out.append(
            {
                "Metric": metric_name,
                "Seed 0 difference (pp)": f"{seed_diffs['0']:.2f}",
                "Seed 42 difference (pp)": f"{seed_diffs['42']:.2f}",
                "Seed 3407 difference (pp)": f"{seed_diffs['3407']:.2f}",
                "Mean difference (pp)": f"{mean_diff:.2f}",
                "t": f"{t_stat:.3f}",
                "df": str(df),
                "p": f"{p_value:.3f}",
                "95% CI (pp)": f"[{ci_low:.2f}, {ci_high:.2f}]",
                "Cohen's dz": f"{cohen_dz:.2f}",
            }
        )
    return out


def build_standard_table(
    seed_rows: Sequence[Dict[str, str]],
    source_table: str,
    order: Sequence[str],
    dataset: Optional[str] = None,
) -> List[Dict[str, object]]:
    out = []
    for name in order:
        selected = select_runs(seed_rows, source_table, name, dataset)
        require_three(selected, f"{source_table} {name}")
        out.append(standard_table_row(name, summarize_runs(selected)))
    return out


def build_table4(seed_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = build_standard_table(seed_rows, "Table 4", TABLE4_ORDER, "PVEL-AD")
    for row in rows:
        row["Params (M)"] = TABLE4_PARAMS_M[row["method"]]
        row["GFLOPs (G)"] = ""
        row["FPS"] = ""
    return rows


def build_table5(seed_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = build_standard_table(seed_rows, "Table 5", TABLE5_ORDER, "PVEL-AD")
    for row in rows:
        row["Params (M)"] = TABLE5_PARAMS_M[row["method"]]
        row["GFLOPs (G)"] = ""
        row["FPS"] = ""
    return rows


def build_table6(seed_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out = []
    for label, source_table, config in TABLE6_SOURCES:
        selected = select_runs(seed_rows, source_table, config, "PVEL-AD")
        require_three(selected, f"Table 6 {label}")
        summary = summarize_runs(selected)
        row = standard_table_row(label, summary)
        row["Params (M)"] = ""
        row["GFLOPs (G)"] = ""
        row["FPS"] = ""
        out.append(row)
    return out


def class_mean_rows(class_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        r
        for r in class_rows
        if r.get("dataset") == "PVEL-AD"
        and r.get("source_table") == "Table 7 / Table 9 / Table 10 support"
        and r.get("experiment_group") == "Sequential component ablation"
        and r.get("aggregation") == "mean_sd_over_3_runs"
    ]


def class_lookup(
    rows: Sequence[Dict[str, str]],
    config: str,
) -> Dict[str, Dict[str, str]]:
    selected = [r for r in rows if r.get("model_or_config") == config]
    if not selected:
        raise ValueError(f"No class-wise rows found for configuration: {config}")
    return {r["class_name"]: r for r in selected}


def build_table9(class_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = class_mean_rows(class_rows)
    baseline = class_lookup(rows, "Baseline")
    full = class_lookup(rows, "Full model")

    classes = list(baseline.keys())
    out = []
    for cls in classes:
        b = baseline[cls]
        g = full[cls]
        out.append(
            {
                "Class": cls,
                "RT-DETR-r18 Precision (%)": f"{float(b['precision_pct']):.2f} ± {float(b['precision_sd_pct']):.2f}",
                "RT-DETR-r18 Recall (%)": f"{float(b['recall_pct']):.2f} ± {float(b['recall_sd_pct']):.2f}",
                "RT-DETR-r18 AP50 (%)": f"{float(b['map50_pct']):.2f} ± {float(b['map50_sd_pct']):.2f}",
                "RT-DETR-GMI Precision (%)": f"{float(g['precision_pct']):.2f} ± {float(g['precision_sd_pct']):.2f}",
                "RT-DETR-GMI Recall (%)": f"{float(g['recall_pct']):.2f} ± {float(g['recall_sd_pct']):.2f}",
                "RT-DETR-GMI AP50 (%)": f"{float(g['map50_pct']):.2f} ± {float(g['map50_sd_pct']):.2f}",
            }
        )
    return out


def build_table10(class_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = class_mean_rows(class_rows)

    lookups = {}
    required_configs = {"Baseline", "GCS Backbone", "GCS + MSLA-AIFI",
                        "GCS + MSLA-AIFI + MPMF-Net", "Full model"}
    for config in required_configs:
        lookups[config] = class_lookup(rows, config)

    classes = list(lookups["Baseline"].keys())
    out = []

    for cls in classes:
        record = {"Class": cls}
        for label, previous, current in TABLE10_STEPS:
            prev_ap = float(lookups[previous][cls]["map50_pct"])
            curr_ap = float(lookups[current][cls]["map50_pct"])
            record[f"{label} ΔAP50 (pp)"] = f"{curr_ap - prev_ap:+.2f}"
        out.append(record)

    return out


def build_f1_audit(seed_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}

    for row in seed_rows:
        key = (
            row.get("dataset", ""),
            row.get("source_table", ""),
            row.get("model_or_config", ""),
        )
        grouped.setdefault(key, []).append(row)

    for (dataset, source_table, config), rows in sorted(grouped.items()):
        if len(rows) != 3:
            continue

        stored_class = [to_float(r.get("class_avg_f1_pct")) for r in rows]
        if all(v is None for v in stored_class):
            class_mean_sd = ""
        else:
            cm, cs = mean_sd(stored_class)
            class_mean_sd = fmt_mean_sd(cm, cs)

        manuscript_vals = [
            manuscript_f1(to_float(r.get("precision_pct")), to_float(r.get("recall_pct")))
            for r in rows
        ]
        mm, ms = mean_sd(manuscript_vals)

        stored_manuscript = [to_float(r.get("manuscript_f1_pct")) for r in rows]
        max_abs_diff = 0.0
        comparable = False
        for calc, stored in zip(manuscript_vals, stored_manuscript):
            if calc is not None and stored is not None:
                comparable = True
                max_abs_diff = max(max_abs_diff, abs(calc - stored))

        out.append(
            {
                "Dataset": dataset,
                "Source table": source_table,
                "Method/configuration": config,
                "Manuscript F1 Mean ± SD (%)": fmt_mean_sd(mm, ms),
                "Class-averaged F1 Mean ± SD (%)": class_mean_sd,
                "Max |stored manuscript F1 - recalculated F1|": (
                    f"{max_abs_diff:.6f}" if comparable else ""
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate manuscript Tables 3–11 from released CSV results."
    )
    parser.add_argument(
        "--per-seed",
        default="reproducibility/results/per_seed_metrics.csv",
        help="Released per-seed metrics CSV.",
    )
    parser.add_argument(
        "--per-class",
        default="reproducibility/results/per_class_metrics.csv",
        help="Released per-class metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="reproducibility/results/regenerated_tables",
        help="Directory for regenerated tables.",
    )
    args = parser.parse_args()

    per_seed_path = Path(args.per_seed)
    per_class_path = Path(args.per_class)
    output_dir = Path(args.output_dir)

    seed_rows = read_csv(per_seed_path)
    class_rows = read_csv(per_class_path)

    if not seed_rows:
        raise ValueError("per_seed_metrics.csv is empty.")
    if not class_rows:
        raise ValueError("per_class_metrics.csv is empty.")

    required_seed_cols = {
        "dataset", "source_table", "model_or_config", "seed_or_run",
        "precision_pct", "recall_pct", "map50_pct", "map50_95_pct",
    }
    missing = required_seed_cols - set(seed_rows[0].keys())
    if missing:
        raise ValueError(f"Missing required per-seed columns: {sorted(missing)}")

    required_class_cols = {
        "dataset", "source_table", "experiment_group", "model_or_config",
        "aggregation", "class_name", "precision_pct", "precision_sd_pct",
        "recall_pct", "recall_sd_pct", "map50_pct", "map50_sd_pct",
    }
    missing = required_class_cols - set(class_rows[0].keys())
    if missing:
        raise ValueError(f"Missing required per-class columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 3
    table3 = build_table3(seed_rows)
    write_csv(
        output_dir / "table3_regenerated.csv",
        table3,
        (
            "Metric",
            "Seed 0 difference (pp)",
            "Seed 42 difference (pp)",
            "Seed 3407 difference (pp)",
            "Mean difference (pp)",
            "t",
            "df",
            "p",
            "95% CI (pp)",
            "Cohen's dz",
        ),
    )

    common_fields = (
        "method",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "mAP50 (%)",
        "mAP50:95 (%)",
        "Params (M)",
        "GFLOPs (G)",
        "FPS",
    )

    # Tables 4–8
    table4 = build_table4(seed_rows)
    table5 = build_table5(seed_rows)
    table6 = build_table6(seed_rows)
    table7 = build_standard_table(seed_rows, "Table 7", TABLE7_ORDER, "PVEL-AD")
    table8 = build_standard_table(seed_rows, "Table 8", TABLE8_ORDER, "PVEL-AD")

    write_csv(output_dir / "table4_regenerated.csv", table4, common_fields)
    write_csv(output_dir / "table5_regenerated.csv", table5, common_fields)
    write_csv(output_dir / "table6_regenerated.csv", table6, common_fields)
    write_csv(output_dir / "table7_regenerated.csv", table7, common_fields)
    write_csv(output_dir / "table8_regenerated.csv", table8, common_fields)

    # Tables 9–10
    table9 = build_table9(class_rows)
    write_csv(
        output_dir / "table9_regenerated.csv",
        table9,
        (
            "Class",
            "RT-DETR-r18 Precision (%)",
            "RT-DETR-r18 Recall (%)",
            "RT-DETR-r18 AP50 (%)",
            "RT-DETR-GMI Precision (%)",
            "RT-DETR-GMI Recall (%)",
            "RT-DETR-GMI AP50 (%)",
        ),
    )

    table10 = build_table10(class_rows)
    table10_fields = ["Class"] + [f"{label} ΔAP50 (pp)" for label, _, _ in TABLE10_STEPS]
    write_csv(output_dir / "table10_regenerated.csv", table10, table10_fields)

    # Table 11
    table11 = build_standard_table(
        seed_rows,
        "Table 11",
        ("RT-DETR-r18", "RT-DETR-GMI"),
        "PV-Multi-Defect",
    )
    write_csv(output_dir / "table11_regenerated.csv", table11, common_fields)

    # Explicit F1 audit requested by Reviewer #4.
    f1_audit = build_f1_audit(seed_rows)
    write_csv(
        output_dir / "f1_aggregation_audit.csv",
        f1_audit,
        (
            "Dataset",
            "Source table",
            "Method/configuration",
            "Manuscript F1 Mean ± SD (%)",
            "Class-averaged F1 Mean ± SD (%)",
            "Max |stored manuscript F1 - recalculated F1|",
        ),
    )

    print(f"Input per-seed file : {per_seed_path}")
    print(f"Input per-class file: {per_class_path}")
    print(f"Output directory     : {output_dir}")
    print()
    print("Generated manuscript tables:")
    for n in range(3, 12):
        print(f"  Table {n}: {output_dir / f'table{n}_regenerated.csv'}")
    print(f"  F1 audit: {output_dir / 'f1_aggregation_audit.csv'}")
    print()
    print("Key checks:")
    for row in table3:
        print(
            f"  Table 3 {row['Metric']}: mean diff={row['Mean difference (pp)']} pp, "
            f"p={row['p']}, 95% CI={row['95% CI (pp)']}"
        )
    for row in table11:
        print(f"  Table 11 {row['method']}: F1={row['F1 (%)']}")


if __name__ == "__main__":
    main()
