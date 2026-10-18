"""Verify the complete Baseline and MSLA-only detectors and their encoders.

Run from the repository root:
    python tools/verify_params.py
    python tools/verify_params.py --full-structure

The default is the eight-class setting used in Table 7. Both models are built
from the repository YAML files without loading weights or starting training.
Total parameter counts are reported before and after repository model fusion.
"""

import argparse
from pathlib import Path
import sys


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
# Use this checkout, including when the script is launched outside the repo root.
sys.path.insert(0, str(REPO_ROOT))

MODEL_SPECS = (
    ("Baseline RT-DETR-r18", "rtdetr-r18.yaml", "AIFI", 789760, 19.88),
    ("MSLA-only", "rtdetr-AIFI-MSLA.yaml", "TransformerEncoderLayer_MSLA", 619588, 19.71),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc", type=int, default=8, help="Number of classes (default: 8).")
    parser.add_argument(
        "--full-structure", action="store_true",
        help="Also print the complete nested PyTorch structure of each unfused model.",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count (default: 4).")
    args = parser.parse_args()
    if args.nc < 1 or args.threads < 1:
        parser.error("--nc and --threads must be positive integers")
    return args


def count_parameters(module):
    """Count all parameters, including parameters with requires_grad=False."""
    return sum(parameter.numel() for parameter in module.parameters())


def check(label, measured, expected):
    passed = measured == expected
    print(f"{'PASS' if passed else 'FAIL'}: {label}: measured={measured}, expected={expected}")
    return passed


def main():
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics.nn.tasks import RTDETRDetectionModel

    torch.set_num_threads(args.threads)
    print(f"Ultralytics: {ultralytics.__version__}; PyTorch: {torch.__version__}", flush=True)
    print(f"Classes: {args.nc}; parameter counting device: CPU", flush=True)
    print("The YAML nc value is overridden in memory; source YAML files are unchanged.", flush=True)
    results = []

    for name, filename, encoder_name, expected_encoder, paper_fused_m in MODEL_SPECS:
        yaml_path = REPO_ROOT / "ultralytics" / "cfg" / "model" / filename
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
        print(f"\n{'=' * 72}\n{name}\nYAML: {yaml_path.relative_to(REPO_ROOT)}", flush=True)
        # verbose=True prints the layer table and repository model summary.
        # eval() is required before folding batch normalization into convolutions.
        model = RTDETRDetectionModel(cfg=str(yaml_path), ch=3, nc=args.nc, verbose=True).eval()
        encoders = [layer for layer in model.model if type(layer).__name__ == encoder_name]
        if len(encoders) != 1:
            raise RuntimeError(f"Expected one {encoder_name} layer, found {len(encoders)}")
        encoder = encoders[0]
        encoder_count = count_parameters(encoder)
        unfused_count = count_parameters(model)
        if args.full_structure:
            print("\nComplete nested structure (unfused):", flush=True)
            print(model, flush=True)
        print(f"Encoder: {type(encoder).__module__}.{encoder_name}", flush=True)
        print(f"Encoder parameters: {encoder_count:,}", flush=True)
        print(f"Unfused total: {unfused_count:,} ({unfused_count / 1e6:.2f}M)", flush=True)

        model.fuse(verbose=True)
        fused_count = count_parameters(model)
        print(f"Fused total: {fused_count:,} ({fused_count / 1e6:.2f}M)", flush=True)
        results.append({
            "name": name, "encoder": encoder_count, "unfused": unfused_count,
            "fused": fused_count, "expected_encoder": expected_encoder, "paper_fused_m": paper_fused_m,
        })
        del encoder, encoders, model

    print("\nParameter comparison (all parameters, not only trainable parameters):")
    print(f"{'Model':<24} {'Encoder':>12} {'Unfused total':>16} {'Fused total':>16}")
    for result in results:
        print(f"{result['name']:<24} {result['encoder']:>12,} "
              f"{result['unfused']:>16,} {result['fused']:>16,}")
    print("\nChecks:")
    checks = []
    for result in results:
        checks.append(check(f"{result['name']} encoder", result["encoder"], result["expected_encoder"]))
    expected_reduction = 170172
    for key in ("encoder", "unfused", "fused"):
        reduction = results[0][key] - results[1][key]
        checks.append(check(f"Baseline minus MSLA-only ({key})", reduction, expected_reduction))
    if args.nc == 8:
        for result in results:
            measured_m = round(result["fused"] / 1e6, 2)
            checks.append(check(f"Table 7 {result['name']} fused Params (M)",
                                measured_m, result["paper_fused_m"]))
    else:
        print("Table 7 rounded-total checks are skipped: Table 7 uses 8 classes.")
    if not all(checks):
        print("Parameter verification FAILED.", file=sys.stderr)
        return 1
    print("Parameter verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
