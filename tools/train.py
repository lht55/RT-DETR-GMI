"""Train an RT-DETR model using this repository's local source code.

Examples (run from the repository root):
    python tools/train.py --model ultralytics/cfg/model/RT-DETR-GMI.yaml --data path/to/data.yaml --seed 0
    python tools/train.py --cfg path/to/train_args.yaml --model ultralytics/cfg/model/rtdetr-r18.yaml --data path/to/data.yaml --seed 42
    python tools/train.py --data path/to/data.yaml --dry-run

Priority for a fresh run: explicit CLI arguments > --cfg YAML > PAPER_DEFAULTS.
--model selects the architecture/checkpoint, --data selects the dataset YAML,
and --cfg selects training hyperparameters (including archived args.yaml files).
IoU-loss selection still follows reproducibility/loss_switching.md; choosing a
model YAML does not change the source-level loss implementation.
"""

import argparse
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Principal PVEL-AD protocol. PV-Multi-Defect runs additionally use --cache.
PAPER_DEFAULTS = {
    "model": "ultralytics/cfg/model/RT-DETR-GMI.yaml",
    "epochs": 300, "patience": 40, "batch": 16, "imgsz": 640,
    "optimizer": "AdamW", "lr0": 0.0001, "lrf": 1.0,
    "momentum": 0.9, "weight_decay": 0.0001,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "seed": 0, "deterministic": True, "amp": False, "cache": False,
    "workers": 4, "pretrained": True, "rect": False, "cos_lr": False,
    "close_mosaic": 0, "nbs": 64, "val": True, "split": "val",
    "iou": 0.7, "max_det": 300, "half": False, "plots": True,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
    "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "project": "runs/train", "name": "exp",
}


def cache_value(value):
    value = value.lower()
    if value in ("true", "false"):
        return value == "true"
    if value in ("ram", "disk"):
        return value
    raise argparse.ArgumentTypeError("cache must be true, false, ram, or disk")


def pretrained_value(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def make_parser():
    # Suppressed defaults prevent absent CLI flags from replacing --cfg values.
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cfg", help="Training YAML or archived args.yaml; CLI values override it.")
    parser.add_argument("--model", help="Model architecture YAML or local .pt checkpoint.")
    parser.add_argument("--data", help="Dataset YAML containing the train/val paths and class names.")
    parser.add_argument("--pretrained", type=pretrained_value,
                        help="Optional checkpoint path, or true/false framework flag.")
    parser.add_argument("--resume", nargs="?", const=True,
                        help="Resume checkpoint path, or use --resume with --model last.pt.")
    parser.add_argument("--device", help="CUDA device such as 0, 0,1, or cpu.")
    parser.add_argument("--optimizer", help="Optimizer name, for example AdamW or SGD.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Run name; use different names for different seeds.")
    for option in ("imgsz", "epochs", "batch", "workers", "seed", "patience", "nbs", "close-mosaic", "max-det"):
        parser.add_argument(f"--{option}", type=int)
    for option in ("lr0", "lrf", "momentum", "weight-decay", "warmup-momentum", "warmup-bias-lr", "iou"):
        parser.add_argument(f"--{option}", type=float)
    parser.add_argument("--warmup-epochs", type=float,
                        help="Forwarded unchanged; this checkout uses a global batch-index limit, not epochs.")
    parser.add_argument("--cache", nargs="?", const=True, type=cache_value,
                        help="Enable RAM caching, or specify true, false, ram, or disk.")
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=False)
    for option in ("amp", "deterministic", "cos-lr", "rect", "val", "plots", "exist-ok"):
        parser.add_argument(f"--{option}", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved arguments without building a model or starting training.")
    return parser


def local_file(value, label, model=False):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty local file path")
    path = Path(value).expanduser()
    candidates = [path, REPO_ROOT / path]
    if model and path.name == value:
        candidates.append(REPO_ROOT / "ultralytics/cfg/model" / path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"{label} not found: {value}")


def resolve_settings(args):
    cli = vars(args).copy()
    dry_run = cli.pop("dry_run", False)
    config_path = cli.pop("cfg", None)
    config = {}
    if config_path:
        with open(local_file(config_path, "Training configuration"), encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("Training configuration must be a YAML mapping")
        if "backbone" in config or "head" in config:
            raise ValueError("An architecture YAML belongs in --model; --cfg is for training parameters")
        # Archived runtime metadata must not redirect a fresh run's output or
        # trigger a second config-file load inside Model.train().
        config.pop("save_dir", None)
        config.pop("cfg", None)
    settings = {**PAPER_DEFAULTS, **config, **cli}
    if not settings.get("data"):
        raise ValueError("Provide --data DATA.yaml or a data field in --cfg")
    if settings.get("task", "detect") != "detect" or settings.get("mode", "train") != "train":
        raise ValueError("This entry point supports task=detect and mode=train only")
    settings.update(task="detect", mode="train")

    # Check option names before constructing the potentially large detector.
    with open(REPO_ROOT / "ultralytics/cfg/default.yaml", encoding="utf-8") as stream:
        valid_keys = yaml.safe_load(stream)
    unknown = set(settings) - set(valid_keys)
    if unknown:
        raise ValueError(f"Unknown training arguments: {', '.join(sorted(map(str, unknown)))}")

    resume = settings.get("resume", False)
    if resume:
        # Match the checkpoint used to build RTDETR to the one being resumed.
        checkpoint = settings["model"] if resume is True else resume
        checkpoint = local_file(checkpoint, "Resume checkpoint", model=True)
        if Path(checkpoint).suffix.lower() != ".pt":
            raise ValueError("Resume requires a local .pt checkpoint")
        settings["model"] = settings["resume"] = checkpoint
        if "pretrained" in cli or ("pretrained" in config and isinstance(settings.get("pretrained"), str)):
            raise ValueError("Resume already restores weights; remove the separate pretrained setting")
    else:
        settings["model"] = local_file(settings["model"], "Model", model=True)
    settings["data"] = local_file(settings["data"], "Dataset YAML")
    if Path(settings["model"]).suffix.lower() not in (".yaml", ".yml", ".pt"):
        raise ValueError("--model must be a .yaml, .yml, or .pt file")
    if Path(settings["data"]).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("--data must be a dataset .yaml or .yml file")
    if not resume and isinstance(settings.get("pretrained"), str):
        settings["pretrained"] = local_file(settings["pretrained"], "Pretrained checkpoint")
        if Path(settings["pretrained"]).suffix.lower() != ".pt":
            raise ValueError("A pretrained checkpoint must be a .pt file")
    return settings, dry_run


def main(argv=None):
    parser = make_parser()
    try:
        settings, dry_run = resolve_settings(parser.parse_args(argv))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print("Resolved training settings (fresh-run priority: CLI > --cfg > script defaults):", flush=True)
    print(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), flush=True)
    print("warmup_epochs is passed unchanged; trainer.py treats it as a global batch-index limit.", flush=True)
    print("IoU loss is selected in ultralytics/models/utils/loss.py; see reproducibility/loss_switching.md.", flush=True)
    if settings.get("resume"):
        print("Resume restores checkpoint arguments. This trainer permits imgsz/batch overrides on resume.", flush=True)
    if dry_run:
        print("Dry run complete: no model was constructed and no training was started.")
        return 0

    # Delayed import lets --help and --dry-run work without loading PyTorch.
    from ultralytics import RTDETR
    from ultralytics.cfg import get_cfg

    get_cfg(overrides=settings.copy())  # Framework type/value validation.
    model = RTDETR(settings["model"])
    if not settings.get("resume") and isinstance(settings.get("pretrained"), str):
        model.load(settings["pretrained"])
    model.train(**settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
