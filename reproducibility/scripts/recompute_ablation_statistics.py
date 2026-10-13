#!/usr/bin/env python3
"""Recompute RT-DETR-GMI ablation statistics using the two companion CSVs.

Python 3.9+; standard library only.
Run from the repository root:
    python reproducibility/scripts/recompute_ablation_statistics.py --check-stored-summary
Inputs use percentage points (0..100). SD and variance use n-1.
Class-average F1 is retained from each run's exported all row.
Manuscript F1 is recomputed separately within each run from its P and R.
Inputs default to ../results relative to this script.
Outputs default to reproducibility/results/recomputed_ablation.
The input defaults also work when launched from another working directory.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re
import statistics as stats

SEEDS = (0, 42, 3407)
FLAGS = {
    "Baseline": (0, 0, 0, 0),
    "GCS Backbone": (1, 0, 0, 0),
    "MSLA-AIFI": (0, 1, 0, 0),
    "MPMF-Net": (0, 0, 1, 0),
    "Inner-MPDIoU": (0, 0, 0, 1),
    "GCS + MSLA-AIFI": (1, 1, 0, 0),
    "GCS + MSLA-AIFI + MPMF-Net": (1, 1, 1, 0),
    "Full model": (1, 1, 1, 1),
}
CLASS_NAMES = (
    "black_core", "crack", "finger", "horizontal_dislocation",
    "short_circuit", "star_crack", "thick_line", "vertical_dislocation",
)
CLASS_METRICS = ("precision", "recall", "f1_score", "map50", "map75", "map50_95")
OVERALL_METRICS = (
    "precision", "recall", "class_avg_f1", "manuscript_f1",
    "map50", "map75", "map50_95", "end_to_end_fps",
)
SEQUENCE = (
    "Baseline", "GCS Backbone", "GCS + MSLA-AIFI",
    "GCS + MSLA-AIFI + MPMF-Net", "Full model",
)
ALL_LABEL = "all(平均数据)"
GROUP = "Sequential component ablation"

def fmt(value, places=2):
    quantum = Decimal(1).scaleb(-places)
    return f"{Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP):.{places}f}"

def mean(values):
    return stats.mean(values)

def sd(values):
    return stats.stdev(values)

def cell(values, places=2):
    return f"{fmt(mean(values), places)} ± {fmt(sd(values), places)}"

def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
        *["| " + " | ".join(str(c) for c in row) + " |" for row in rows],
    ])

def overall_metric(runs, index):
    return [run["overall"][index] for run in runs]

def class_values(runs, cls, index):
    return [run["classes"][cls][index] for run in runs]

def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field.lstrip("\ufeff") for field in (reader.fieldnames or [])]
        reader.fieldnames = fields
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path.name}: duplicate column names")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed CSV row")
    return rows

def number(value, context, percentage=False, positive=False):
    try:
        result = Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        raise ValueError(f"{context}: missing or invalid number {value!r}") from None
    if not result.is_finite():
        raise ValueError(f"{context}: non-finite number")
    if percentage and not Decimal(0) <= result <= Decimal(100):
        raise ValueError(f"{context}: percentage outside 0..100")
    if positive and result <= 0:
        raise ValueError(f"{context}: expected a positive number")
    return result

def parse_seed(value, context):
    seed = number(value, context)
    if seed != seed.to_integral_value() or int(seed) not in SEEDS:
        raise ValueError(f"{context}: expected seed 0, 42 or 3407")
    return int(seed)

def load_data(class_path, seed_path):
    class_fields = [
        "dataset", "experiment_group", "model_or_config", "aggregation",
        "seed_or_run", "class_name",
        *[key + "_pct" for key in CLASS_METRICS],
        *[key + "_sd_pct" for key in CLASS_METRICS],
    ]
    rows = read_csv(class_path, class_fields)
    raw, stored = {}, {}
    for row in rows:
        if row["experiment_group"] != GROUP:
            continue
        name, cls = row["model_or_config"], row["class_name"]
        if name not in FLAGS or cls not in CLASS_NAMES or row["dataset"] != "PVEL-AD":
            raise ValueError(f"Unexpected ablation class row: {name}/{cls}")
        if row["aggregation"] == "mean_sd_over_3_runs":
            key = (name, cls)
            if key in stored:
                raise ValueError(f"Duplicate stored summary: {key}")
            stored[key] = row
        elif row["aggregation"] == "single_run":
            seed = parse_seed(row["seed_or_run"], f"{name}/{cls}")
            key = (name, seed, cls)
            if key in raw:
                raise ValueError(f"Duplicate raw class row: {key}")
            raw[key] = [
                number(row[metric + "_pct"], f"{key}/{metric}", percentage=True)
                for metric in CLASS_METRICS
            ]
        else:
            raise ValueError(f"Unexpected aggregation: {row['aggregation']!r}")
    expected = {(name, seed, cls) for name in FLAGS for seed in SEEDS for cls in CLASS_NAMES}
    if set(raw) != expected:
        raise ValueError(f"Class seed coverage mismatch: missing {sorted(expected-set(raw))}; extra {sorted(set(raw)-expected)}")

    overall_fields = [
        "dataset", "split", "model_or_config", "seed_or_run", "precision_pct",
        "recall_pct", "class_avg_f1_pct", "manuscript_f1_pct", "map50_pct",
        "map75_pct", "map50_95_pct", "params_count", "params_m", "gflops_g",
        "end_to_end_fps",
    ]
    totals = {}
    for row in read_csv(seed_path, overall_fields):
        name = row["model_or_config"]
        if name not in FLAGS or row["dataset"] != "PVEL-AD" or row["split"] != "test":
            raise ValueError(f"Unexpected overall evaluation row: {name}")
        seed = parse_seed(row["seed_or_run"], name)
        key = (name, seed)
        if key in totals:
            raise ValueError(f"Duplicate overall seed row: {key}")
        overall = [
            number(row[k], f"{key}/{k}", percentage=True)
            for k in ("precision_pct", "recall_pct", "class_avg_f1_pct",
                      "map50_pct", "map75_pct", "map50_95_pct")
        ]
        p, r = overall[:2]
        f1 = 2 * p * r / (p + r) if p + r else Decimal(0)
        saved_f1 = number(row["manuscript_f1_pct"], f"{key}/manuscript_f1_pct", percentage=True)
        if abs(saved_f1 - f1) > Decimal("0.000000005"):
            raise ValueError(f"{key}: stored manuscript F1 does not match per-run P/R")
        params = number(row["params_count"], f"{key}/params_count", positive=True)
        if params != params.to_integral_value():
            raise ValueError(f"{key}: params_count must be an integer")
        if number(row["params_m"], f"{key}/params_m", positive=True) != Decimal(fmt(params / 1_000_000)):
            raise ValueError(f"{key}: params_m differs from rounded params_count")
        classes = {cls: raw[(name, seed, cls)] for cls in CLASS_NAMES}
        # Printed per-class and all metrics may differ slightly due to export rounding.
        for i in range(6):
            class_average = mean([values[i] for values in classes.values()])
            if abs(class_average - overall[i]) > Decimal("0.01"):
                raise ValueError(f"{key}: all metric {i} inconsistent with class values")
        classes[ALL_LABEL] = overall
        totals[key] = {
            "seed": seed, "classes": classes, "overall": overall,
            "manuscript_f1": f1, "params": int(params),
            "gflops": number(row["gflops_g"], f"{key}/gflops", positive=True),
            "fps": number(row["end_to_end_fps"], f"{key}/fps", positive=True),
        }
    expected_totals = {(name, seed) for name in FLAGS for seed in SEEDS}
    if set(totals) != expected_totals:
        raise ValueError(f"Overall seed coverage mismatch: missing {sorted(expected_totals-set(totals))}")
    data = {name: [totals[(name, seed)] for seed in SEEDS] for name in FLAGS}
    for name, runs in data.items():
        if len({r["params"] for r in runs}) != 1 or len({r["gflops"] for r in runs}) != 1:
            raise ValueError(f"{name}: Params/GFLOPs differ across seeds")
    return data, stored

def check_stored_summary(data, stored):
    expected = {(name, cls) for name in data for cls in CLASS_NAMES}
    if set(stored) != expected:
        raise ValueError("Stored class summaries do not cover all 64 configuration/class pairs")
    for name, cls in expected:
        for index, metric in enumerate(CLASS_METRICS):
            values = class_values(data[name], cls, index)
            for suffix, computed in (("_pct", mean(values)), ("_sd_pct", sd(values))):
                key = metric + suffix
                if number(stored[(name, cls)][key], f"{name}/{cls}/{key}") != Decimal(fmt(computed)):
                    raise ValueError(f"Stored summary mismatch: {name}/{cls}/{key}")

def metric_triplet(values):
    return [fmt(mean(values)), fmt(sd(values)), fmt(stats.variance(values))]

def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

def write_summaries(data, out):
    class_headers = ["dataset", "model_or_config", "class_name", "n_runs", "seeds"]
    class_headers += [
        metric + suffix for metric in CLASS_METRICS
        for suffix in ("_mean_pct", "_sd_pp", "_variance_pp2")
    ]
    class_rows = [
        ["PVEL-AD", name, cls, 3, "0;42;3407",
         *[v for i in range(6) for v in metric_triplet(class_values(runs, cls, i))]]
        for name, runs in data.items() for cls in CLASS_NAMES
    ]
    write_csv(out / "ablation_class_summary.csv", class_headers, class_rows)

    overall_headers = ["dataset", "split", "model_or_config", "n_runs", "seeds"]
    for metric in OVERALL_METRICS:
        overall_headers += (
            [metric + "_mean", metric + "_sd", metric + "_variance"]
            if metric == "end_to_end_fps" else
            [metric + "_mean_pct", metric + "_sd_pp", metric + "_variance_pp2"]
        )
    overall_headers += ["params_count", "params_m", "gflops_g", "fps_mean_integer", "fps_sd_integer"]
    overall_rows = []
    for name, runs in data.items():
        columns = [
            overall_metric(runs, 0), overall_metric(runs, 1), overall_metric(runs, 2),
            [r["manuscript_f1"] for r in runs],
            *[overall_metric(runs, i) for i in (3, 4, 5)], [r["fps"] for r in runs],
        ]
        overall_rows.append([
            "PVEL-AD", "test", name, 3, "0;42;3407",
            *[v for values in columns for v in metric_triplet(values)],
            runs[0]["params"], fmt(Decimal(runs[0]["params"]) / 1_000_000),
            fmt(runs[0]["gflops"], 1), fmt(mean(columns[-1]), 0), fmt(sd(columns[-1]), 0),
        ])
    write_csv(out / "ablation_overall_summary.csv", overall_headers, overall_rows)

    comparison_headers = ["class_name"]
    for label in ("baseline", "full_model"):
        comparison_headers += [
            label + "_" + metric + suffix for metric in ("precision", "recall", "ap50")
            for suffix in ("_mean_pct", "_sd_pp")
        ]
    comparison_rows = []
    for cls in CLASS_NAMES:
        row = [cls]
        for name in ("Baseline", "Full model"):
            for i in (0, 1, 3):
                values = class_values(data[name], cls, i)
                row.extend([fmt(mean(values)), fmt(sd(values))])
        comparison_rows.append(row)
    write_csv(out / "table9_per_class_comparison.csv", comparison_headers, comparison_rows)

    delta_headers = [
        "class_name", "baseline_ap50_mean_pct", "gcs_ap50_mean_pct",
        "gcs_msla_ap50_mean_pct", "gcs_msla_mpmf_ap50_mean_pct", "full_ap50_mean_pct",
        "gcs_delta_pp", "msla_delta_pp", "mpmf_delta_pp", "inner_mpdiou_delta_pp",
    ]
    delta_rows = []
    for cls in CLASS_NAMES:
        # Table 10 uses displayed AP50 means (2 dp), then adjacent subtraction.
        rounded = [Decimal(fmt(mean(class_values(data[name], cls, 3)))) for name in SEQUENCE]
        differences = [rounded[i+1] - rounded[i] for i in range(4)]
        delta_rows.append([cls, *[fmt(x) for x in rounded], *[fmt(x) for x in differences]])
    write_csv(out / "table10_incremental_ap50.csv", delta_headers, delta_rows)



def write_markdown(DATA, OUTPUT):
    parts = []

    def add(text):
        parts.append(text.strip())

    def aggregate_columns(runs):
        return [overall_metric(runs, 0), overall_metric(runs, 1), overall_metric(runs, 2), [r['manuscript_f1'] for r in runs], *[overall_metric(runs, i) for i in (3, 4, 5)]]

    def structural_columns(runs):
        return [fmt(Decimal(runs[0]['params']) / 1000000), fmt(runs[0]['gflops'], 1)]

    def signed(value, places=2):
        return ('+' if value >= 0 else '') + fmt(value, places)
    HEADERS = ['Model setting', 'Precision/%', 'Recall/%', '类别平均 F1/%', '论文口径 F1/%', 'mAP50/%', 'mAP75/%', 'mAP50:95/%']
    add('# Ablation Experiment Statistics with Three Random Seeds\n\n8 个消融配置，随机种子为 0、42、3407。Baseline 为 RT-DETR-r18，Full model 为 RT-DETR-GMI。')
    add('## 1. 计算方法\n\n对三个 seed 的结果计算均值、样本方差和样本标准差：\n\n$$\n\\bar{x}=\\frac{x_0+x_{42}+x_{3407}}{3},\\qquad\ns^2=\\frac{\\sum_{i\\in\\{0,42,3407\\}}(x_i-\\bar{x})^2}{2},\\qquad\nSD=\\sqrt{s^2}.\n$$\n\n检测指标以百分比展示；SD 为百分点，方差为百分点平方（pp²）。FPS 方差为 FPS²。主表 FPS 取整数，明细保留两位小数。\n\n整体表保留两种 F1：**类别平均 F1**为各类别 F1 的平均值；**论文口径 F1**为每个 seed 按整体 P、R 计算 $2PR/(P+R)$，再统计 Mean ± SD。类别明细中的 F1 为该类别的 F1。\n')
    add('## 2. 更新后的消融实验主表（Mean ± SD）')
    add(table(['Model setting', 'GCS Backbone', 'MSLA-AIFI', 'MPMF-Net', 'Inner-MPDIoU', 'Precision/%', 'Recall/%', 'mAP50/%', 'mAP50:95/%', 'Params/M', 'GFLOPs/G', 'FPS'], [[name, *['✓' if flag else '—' for flag in FLAGS[name]], *[cell(overall_metric(runs, i)) for i in (0, 1, 3, 5)], *structural_columns(runs), cell([r['fps'] for r in runs], 0)] for name, runs in DATA.items()]))
    add('## 3. 扩展消融表（含 F1 和 mAP75）')
    add(table(HEADERS + ['Params/M', 'GFLOPs/G', 'FPS'], [[name, *[cell(values) for values in aggregate_columns(runs)], *structural_columns(runs), cell([r['fps'] for r in runs])] for name, runs in DATA.items()]))
    add('## 4. 均值表')
    add(table(HEADERS + ['Params/M', 'GFLOPs/G', 'FPS'], [[name, *[fmt(mean(values)) for values in aggregate_columns(runs)], *structural_columns(runs), fmt(mean([r['fps'] for r in runs]))] for name, runs in DATA.items()]))
    add('## 5. 样本方差表\n\n检测指标方差单位为 pp²，FPS 方差单位为 FPS²。')
    add(table(['Model setting', 'Precision', 'Recall', '类别平均 F1', '论文口径 F1', 'mAP50', 'mAP75', 'mAP50:95', 'FPS'], [[name, *[fmt(stats.variance(values)) for values in aggregate_columns(runs)], fmt(stats.variance([r['fps'] for r in runs]))] for name, runs in DATA.items()]))
    add('## 6. 相对基础模型的变化\n\n检测指标变化单位为百分点；按未舍入的均值计算差值后保留两位小数。')
    baseline = DATA['Baseline']
    baseline_means = [mean(values) for values in aggregate_columns(baseline)]
    add(table(['Model setting', 'ΔPrecision', 'ΔRecall', 'Δ类别平均 F1', 'Δ论文口径 F1', 'ΔmAP50', 'ΔmAP50:95', 'ΔParams/M', 'ΔGFLOPs/G', 'ΔFPS'], [[name, *[signed(mean(aggregate_columns(runs)[i]) - baseline_means[i]) for i in (0, 1, 2, 3, 4, 6)], signed(Decimal(runs[0]['params'] - baseline[0]['params']) / 1000000), signed(runs[0]['gflops'] - baseline[0]['gflops'], 1), signed(mean([r['fps'] for r in runs]) - mean([r['fps'] for r in baseline]))] for name, runs in DATA.items() if name != 'Baseline']))
    precision_name = max(DATA, key=lambda name: mean(overall_metric(DATA[name], 0)))
    map50_name = max(DATA, key=lambda name: mean(overall_metric(DATA[name], 3)))
    map95_name = max(DATA, key=lambda name: mean(overall_metric(DATA[name], 5)))
    recall_name = max(DATA, key=lambda name: mean(overall_metric(DATA[name], 1)))
    fps_name = max(DATA, key=lambda name: mean([r['fps'] for r in DATA[name]]))
    overview = []
    if precision_name == map50_name == map95_name:
        name = precision_name
        overview.append(f'- {name} 的 Precision、mAP50 和 mAP50:95 均值最高，分别为 {fmt(mean(overall_metric(DATA[name], 0)))}%、{fmt(mean(overall_metric(DATA[name], 3)))}% 和 {fmt(mean(overall_metric(DATA[name], 5)))}%。')
    else:
        for name, label, i in ((precision_name, 'Precision', 0), (map50_name, 'mAP50', 3), (map95_name, 'mAP50:95', 5)):
            overview.append(f'- {name} 的 {label} 均值最高，为 {fmt(mean(overall_metric(DATA[name], i)))}%。')
    overview.append(f'- {recall_name} 的 Recall 均值最高，为 {fmt(mean(overall_metric(DATA[recall_name], 1)))}%。')
    overview.append(f"- {fps_name} 的 FPS 均值最高，为 {fmt(mean([r['fps'] for r in DATA[fps_name]]))}。")
    add('## 7. 结果概览\n\n' + '\n'.join(overview))
    add('## 8. 各实验设置的详细结果')
    class_headers = ['Class Name', 'Precision/%', 'Recall/%', 'F1-Score/%', 'AP50/%', 'AP75/%', 'AP50:95/%']
    raw_class_count = 0
    overall_count = 0
    for index, (name, runs) in enumerate(DATA.items(), 1):
        add(f'### 8.{index} {name}')
        add('#### 三次随机种子的整体结果')
        overall_rows = []
        for run in runs:
            values = run['overall']
            overall_rows.append([run['seed'], fmt(values[0]), fmt(values[1]), fmt(values[2]), fmt(run['manuscript_f1']), *[fmt(values[i]) for i in (3, 4, 5)], *structural_columns(runs), fmt(run['fps'])])
            overall_count += 1
        add(table(['Seed', *HEADERS[1:], 'Params/M', 'GFLOPs/G', 'FPS'], overall_rows))
        add('#### 每个随机种子的类别结果')
        raw_rows = []
        for run in runs:
            for cls in CLASS_NAMES:
                raw_rows.append([run['seed'], cls, *[fmt(v) for v in run['classes'][cls]]])
                raw_class_count += 1
        add(table(['Seed', *class_headers], raw_rows))
        add('#### 类别级 Mean ± SD')
        add(table(class_headers, [[cls, *[cell(class_values(runs, cls, i)) for i in range(6)]] for cls in (*CLASS_NAMES, ALL_LABEL)]))
        add('#### 类别级样本方差（pp²）')
        add(table(['Class Name', 'Precision', 'Recall', 'F1-Score', 'AP50', 'AP75', 'AP50:95'], [[cls, *[fmt(stats.variance(class_values(runs, cls, i))) for i in range(6)]] for cls in (*CLASS_NAMES, ALL_LABEL)]))
    document = '\n\n'.join(parts) + '\n'
    assert raw_class_count == 192 and overall_count == 24
    assert len(re.findall('^### 8\\.\\d ', document, re.M)) == 8
    assert document.count('#### 每个随机种子的类别结果') == 8
    for forbidden in ('paper_data', 'sha256', '数据来源', '原始文件', '核验', '校验'):
        assert forbidden not in document
    blocks = re.findall('(?:^\\|.*\\n)+', document, re.M)
    for block in blocks:
        rows = block.strip().splitlines()
        lengths = {len(row.split('|')[1:-1]) for row in rows}
        assert len(lengths) == 1, rows
    sections = document.split('### 8.')[1:]
    for (name, runs), section in zip(DATA.items(), sections):
        raw = section.split('#### 每个随机种子的类别结果\n\n', 1)[1].split('\n\n####', 1)[0]
        raw_lines = raw.splitlines()[2:]
        expected = [[str(run['seed']), cls, *[fmt(v) for v in run['classes'][cls]]] for run in runs for cls in CLASS_NAMES]
        actual = [[c.strip() for c in line.split('|')[1:-1]] for line in raw_lines]
        assert actual == expected, name
    OUTPUT.write_text(document, encoding='utf-8')

def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=Path, default=results_dir / "per_class_metrics.csv")
    parser.add_argument("--per-seed", type=Path, default=results_dir / "ablation_per_seed_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=results_dir / "recomputed_ablation")
    parser.add_argument("--check-stored-summary", action="store_true",
                        help="Verify the 64 stored class Mean/SD summaries against raw seeds.")
    args = parser.parse_args()
    try:
        data, stored = load_data(args.per_class, args.per_seed)
        if args.check_stored_summary:
            check_stored_summary(data, stored)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_summaries(data, args.output_dir)
        write_markdown(data, args.output_dir / "ablation_experiment_statistics_recomputed.md")
    except (ValueError, OSError, InvalidOperation) as error:
        parser.error(str(error))
    print("Recomputed 8 configurations, 24 overall runs, and 192 per-class measurements.")
    if args.check_stored_summary:
        print("All 64 stored class summaries match the raw seed calculations.")
    print(f"Outputs: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()

