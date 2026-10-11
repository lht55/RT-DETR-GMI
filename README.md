# RT-DETR-GMI

RT-DETR-GMI is a Python/PyTorch object detection project based on Ultralytics `8.0.201` RT-DETR code with custom model components and configuration.

This repository contains the source code, model configurations, reproducibility documentation, fixed dataset split manifests, and machine-readable numerical results used to support the experiments reported in the revised manuscript. It does not include the original datasets, training images, labels, trained weights, or checkpoints.

## License and Attribution

This project is based on the Ultralytics `8.0.201` codebase, including RT-DETR framework code from https://github.com/ultralytics/ultralytics. The bundled and modified Ultralytics-derived code is released under the GNU Affero General Public License v3.0 (`AGPL-3.0`), consistent with the upstream Ultralytics license.

Original Ultralytics copyright and license notices are retained in the source files. Third-party modules bundled under `ultralytics/` are not claimed as original project code; see `THIRD_PARTY_NOTICES.md` for the current notice and license audit.

## MSLA Attribution

This project uses the MSLA idea with reference to the MSLAU-Net paper (arXiv:2505.18823). The MSLA algorithmic concept is not claimed as original to this project.

The MSLA code shipped here is an independent PyTorch implementation in `ultralytics/nn/extra_modules/msla_reimplementation.py`, written from the paper's public algorithm description, formulas, and structure diagrams. **It is not the experimental implementation used to generate the manuscript's tables.** Its separate branch attentions, AdaptiveScaleFusion, and BatchNorm differ from the historical shared-attention MSLA-AIFI.

The manuscript experiments adapted Lan et al.'s MSLAU-Net implementation to the RT-DETR AIFI encoder, using [Monsoon49/MSLAU-Net](https://github.com/Monsoon49/MSLAU-Net) at commit `efdb058ebe363434376c8ef6cbf689391b60b7fc`, with historical upstream files `networks/msla.py` and `networks/linear_attention.py`. The upstream repository has no explicit software license identified for this dependency, so this repository does not redistribute its original source files. The original encoder is now available as the owner's external local archive; it remains outside this public release.

The current `RT-DETR-GMI.yaml` and `rtdetr-AIFI-MSLA.yaml` resolve to the public reimplementation. They do not, by themselves, reproduce the paper's MSLA results or parameter counts. The external archived encoder has now been instantiated and measured: AIFI 789,760; historical MSLA-AIFI 619,588; reduction 170,172. See [the MSLA reproducibility record](reproducibility/msla/README.md) for archive loading, reconstructed Table 4/5 variants, confirmed implementation details, and the remaining full-experiment inputs.

Please cite the MSLAU-Net paper when discussing or reusing the MSLA idea: https://arxiv.org/abs/2505.18823

## Project Layout

* `RT-DETR-GMI.yaml`: released RT-DETR-GMI model graph using the public MSLA reimplementation; historical MSLA reproduction requires the external experimental encoder.
* `train.py`: training entry point.
* `val.py`: validation and metric reporting entry point.
* `export.py`: model export entry point.
* `heatmap.py`: Grad-CAM visualization helper.
* `main_profile.py`: model profiling helper.
* `ultralytics/`: modified Ultralytics-based framework code required to run RT-DETR-GMI and the released ablation configurations.
* `ultralytics/cfg/model/REPRODUCIBILITY.md`: detailed experimental protocol, model-selection procedure, source-level ablation instructions, evaluation settings, and table-specific reproduction guide.
* `reproducibility/splits/`: fixed train/validation/test split manifests used in the reported experiments.
* `reproducibility/results/`: machine-readable per-seed metrics, per-class metrics, representative confusion matrices, and supporting numerical results.
* `reproducibility/msla/`: historical MSLA experiment specifications and guarded verification interfaces; these are not complete training configurations.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install this project in editable mode if you want local imports and console entry points to use this working tree:

```bash
pip install -e .
```

PyTorch and CUDA versions should match your local GPU driver and CUDA runtime.

This public source set intentionally excludes unrelated experimental modules and CUDA/C++ operators that are not needed by `RT-DETR-GMI.yaml`. KAT-related modules and the local `rational_kat_cu` copy are not part of this minimal release; users who need KAT experiments must install and license-check those external components separately.

## Training

Prepare your dataset separately and pass the dataset YAML path explicitly:

```bash
python train.py --model RT-DETR-GMI.yaml --data path/to/data.yaml
```

Common options:

```bash
python train.py --model RT-DETR-GMI.yaml --data path/to/data.yaml --epochs 300 --batch 16 --device 0
```

Training outputs are written under `runs/` by default and are ignored by Git.

For the exact experimental settings, random seeds, dataset splits, model-selection protocol, and table-specific reproduction instructions used in the manuscript, see:

```text
ultralytics/cfg/model/REPRODUCIBILITY.md
```

## Validation

Validate a trained checkpoint by passing both checkpoint and dataset YAML paths:

```bash
python val.py --model runs/train/exp/weights/best.pt --data path/to/data.yaml --split test
```

`val.py` writes a `paper_data.txt` summary under the validation output directory.

The repository also provides machine-readable numerical results supporting the repeated-run statistics, ablation studies, detector comparisons, per-class analyses, second-dataset evaluation, and representative confusion matrices reported in the revised manuscript. These files are available under:

```text
reproducibility/results/
```

The principal result files include:

```text
reproducibility/results/per_seed_metrics.csv
reproducibility/results/per_class_metrics.csv
reproducibility/results/confusion_matrix_rtdetr_r18_seed0.csv
reproducibility/results/confusion_matrix_rtdetr_gmi_seed0.csv
```

The exact dataset split manifests used for the reported experiments are available under:

```text
reproducibility/splits/
```

This repository does not provide pretrained or trained model weights. Users must provide their own checkpoint paths. Download URLs may be added later after the files and redistribution terms are confirmed.

## Export

Export a trained checkpoint:

```bash
python export.py --model runs/train/exp/weights/best.pt --format onnx
```

Exported files such as `*.onnx` are ignored by Git.

## Heatmap Visualization

Generate a Grad-CAM heatmap with a trained checkpoint and an image or image directory:

```bash
python heatmap.py --weight runs/train/exp/weights/best.pt --source path/to/image_or_dir --output runs/heatmap
```

The Grad-CAM helper requires the `grad-cam` package.

## Reproducibility

The repository provides reproducibility materials supporting the experiments reported in Tables 3–11 of the revised manuscript.

For experiments involving MSLA, the released numerical records describe historical runs. The archived encoder and all seven branch/kernel configurations can now be loaded locally with the new tools; non-final variants are reconstructed from recorded source-edit rules. Complete detector training/metric reproduction still requires the historical model graphs and run settings; no training or metric evaluation was performed in this remediation. See [MSLA reproducibility status](reproducibility/msla/README.md).

The detailed protocol is documented in:

```text
ultralytics/cfg/model/REPRODUCIBILITY.md
```

It includes:

* experimental environment and training settings;
* random seeds `0`, `42`, and `3407`;
* fixed PVEL-AD and PV-Multi-Defect dataset splits;
* validation-based MSLA architecture selection;
* source-level ablation procedures;
* test-set evaluation protocol;
* per-table reproduction instructions;
* machine-readable per-seed and per-class numerical results.

The original image datasets are not redistributed in this repository. Researchers should obtain the corresponding public datasets separately and use the released split manifests to reconstruct the train, validation, and test subsets used in the reported experiments.

## Third-Party Code and License Notes

This project includes an Ultralytics-derived codebase and the MambaOut block used by RT-DETR-GMI. Original copyright and license notices are preserved where present.

The root `LICENSE` contains the AGPL-3.0 license text used by the upstream Ultralytics codebase. Before publishing, confirm the redistribution status of all bundled third-party modules listed in `THIRD_PARTY_NOTICES.md`.
