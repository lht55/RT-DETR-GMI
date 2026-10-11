"""Verify the original AIFI and experimental MSLA-AIFI parameter counts."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_aifi():
    """Load unchanged repository sources without starting the training application."""
    prefix = "_msla_parameter_check"

    for name in (
        prefix,
        f"{prefix}.nn",
        f"{prefix}.nn.modules",
        f"{prefix}.utils",
    ):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    guard = ModuleType(f"{prefix}.utils.torch_utils")
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    guard.TORCH_1_9 = version >= (1, 9)
    sys.modules[guard.__name__] = guard

    source_dir = REPO_ROOT / "ultralytics" / "nn" / "modules"

    load_file(source_dir / "conv.py", f"{prefix}.nn.modules.conv")
    load_file(source_dir / "utils.py", f"{prefix}.nn.modules.utils")

    return load_file(
        source_dir / "transformer.py",
        f"{prefix}.nn.modules.transformer",
    ).AIFI


def main():
    aifi = load_aifi()(c1=256, cm=1024, num_heads=8)

    msla_path = (
        REPO_ROOT
        / "ultralytics"
        / "nn"
        / "extra_modules"
        / "MSLA.py"
    )

    experimental = load_file(msla_path, "_experimental_msla")

    msla_aifi = experimental.TransformerEncoderLayer_MSLA(
        c1=256,
        cm=1024,
        num_heads=8,
    )

    aifi_parameters = sum(p.numel() for p in aifi.parameters())
    msla_parameters = sum(p.numel() for p in msla_aifi.parameters())
    reduction = aifi_parameters - msla_parameters

    print(f"AIFI: {aifi_parameters}")
    print(f"MSLA-AIFI: {msla_parameters}")
    print(
        f"Reduction: {reduction} parameters "
        f"(approximately {reduction / 1e6:.2f}M)"
    )

    if (aifi_parameters, msla_parameters, reduction) != (
        789760,
        619588,
        170172,
    ):
        print(
            "Parameter verification failed: measured counts differ from the references.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
