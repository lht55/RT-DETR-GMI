# RT-DETR-GMI

RT-DETR-GMI is a Python/PyTorch object detection project based on the Ultralytics `8.0.201` RT-DETR codebase with custom model components and experimental configurations.

This repository contains the source code, model configurations, reproducibility documentation, fixed dataset split manifests, and machine-readable numerical results used to support the experiments reported in the revised manuscript. It does not include the original datasets, training images, labels, trained weights, or checkpoints.

## License and Attribution

This project is based on the Ultralytics `8.0.201` codebase, including the RT-DETR framework code from https://github.com/ultralytics/ultralytics.

The bundled and modified Ultralytics-derived code is released under the GNU Affero General Public License v3.0 (`AGPL-3.0`), consistent with the upstream Ultralytics license.

Original Ultralytics copyright and license notices are retained in the source files. Third-party modules bundled under `ultralytics/` are not claimed as original project code. See `THIRD_PARTY_NOTICES.md` for the current attribution and license audit.

## MSLA Attribution

This project uses the MSLA idea with reference to the MSLAU-Net paper (arXiv:2505.18823). The MSLA algorithmic concept is not claimed as original to this project.

The implementation used to generate the manuscript's MSLA results is provided in:

```text
ultralytics/nn/extra_modules/MSLA.py
```

It was extracted from the experimental training source while preserving the MSLA-AIFI parameter definitions and forward computation used in the reported experiments.

The final MSLA configuration uses:

```text
4 depthwise-convolution branches
kernel sizes = (3, 5, 7, 9)
shared LinearAttention
learnable branch scale weights
```

The manuscript experiments adapted Lan et al.'s MSLAU-Net implementation to the RT-DETR AIFI encoder using:

```text
Repository: Monsoon49/MSLAU-Net
Commit: efdb058ebe363434376c8ef6cbf689391b60b7fc
Upstream files:
networks/msla.py
networks/linear_attention.py
```

The experimental RT-DETR encoder adaptation used for the reported manuscript results is included in this repository as `MSLA.py`.

See `THIRD_PARTY_NOTICES.md` for provenance and licensing notes.

The current model configurations:

```text
ultralytics/cfg/model/RT-DETR-GMI.yaml
ultralytics/cfg/model/rtdetr-AIFI-MSLA.yaml
```

resolve to this experimental MSLA implementation.

For:

```text
c1 = 256
cm = 1024
num_heads = 8
```

the encoder parameter counts are:

```text
AIFI:      789,760
MSLA-AIFI: 619,588
Reduction: 170,172
```

This corresponds to an approximately `0.17M` parameter reduction.

See the MSLA reproducibility guide:

```text
ultralytics/nn/extra_modules/README.md
```

for the Table 4/5 branch and kernel variants, implementation provenance, and parameter-count consistency.

Please cite the MSLAU-Net paper when discussing or reusing the MSLA idea:

https://arxiv.org/abs/2505.18823

## Project Layout

The main files relevant to reproducing the manuscript experiments are:

```text
ultralytics/cfg/model/RT-DETR-GMI.yaml
    Final RT-DETR-GMI model configuration.

ultralytics/nn/extra_modules/MSLA.py
    MSLA-AIFI implementation used for the reported experimental results.

ultralytics/nn/extra_modules/README.md
    MSLA provenance, Table 4/5 source variants, and parameter-count verification.

tools/train.py
    RT-DETR training entry point.

tools/val.py
    Validation and test-set metric reporting entry point.

tools/export.py
    Model export entry point.

tools/heatmap.py
    Grad-CAM visualization helper.

tools/verify_params.py
    Parameter-count verification utility for AIFI and MSLA-AIFI.

main_profile.py
    Model profiling helper.

ultralytics/
    Modified Ultralytics-based framework code required to run RT-DETR-GMI and the released ablation configurations.

ultralytics/cfg/model/reproducibility.md
    Main reproducibility guide covering experimental settings, dataset splits, ablations, evaluation, and Tables 3–11.

reproducibility/external_baselines.md
    Framework-specific reproduction information for YOLO and D-FINE external baselines.

reproducibility/loss_switching.md
    Source-level instructions for switching GIoU, Inner-IoU variants, MPDIoU, and Inner-MPDIoU.

reproducibility/splits/
    Fixed train/validation/test split manifests used in the reported experiments.

reproducibility/results/
    Machine-readable per-seed metrics, per-class metrics, confusion matrices, and supporting numerical results.

THIRD_PARTY_NOTICES.md
    Third-party code provenance and licensing notes.
```

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the repository in editable mode so that local imports and the Ultralytics console entry point use this working tree:

```bash
pip install -e .
```

PyTorch and CUDA versions should match the local GPU driver and CUDA runtime.

The main experimental environment used in the manuscript is documented in:

```text
ultralytics/cfg/model/reproducibility.md
```

This public source set intentionally excludes unrelated experimental modules and CUDA/C++ operators that are not required by `RT-DETR-GMI.yaml`.

KAT-related modules and the local `rational_kat_cu` copy are not part of this minimal public release. Researchers requiring those unrelated experimental modules should obtain and license-check them separately.

## Dataset Preparation

The original image datasets are not redistributed in this repository.

Researchers should obtain the corresponding public datasets separately and reconstruct the train, validation, and test subsets using the released split manifests under:

```text
reproducibility/splits/
```

For PVEL-AD, the released manifests are:

```text
reproducibility/splits/pvel_ad_train.txt
reproducibility/splits/pvel_ad_val.txt
reproducibility/splits/pvel_ad_test.txt
```

For PV-Multi-Defect, the released manifests are:

```text
reproducibility/splits/pv_multi_defect_train.txt
reproducibility/splits/pv_multi_defect_val.txt
reproducibility/splits/pv_multi_defect_test.txt
```

The local dataset YAML used for training should reference image sets matching these manifests.

## Training

Prepare the dataset separately and provide the dataset YAML path explicitly.

Example:

```bash
python tools/train.py \
  --model ultralytics/cfg/model/RT-DETR-GMI.yaml \
  --data path/to/data.yaml
```

Example with common options:

```bash
python tools/train.py \
  --model ultralytics/cfg/model/RT-DETR-GMI.yaml \
  --data path/to/data.yaml \
  --epochs 300 \
  --batch 16 \
  --device 0
```

Training outputs are written under `runs/` by default and are ignored by Git.

The reported RT-DETR experiments used three independent random seeds:

```text
0
42
3407
```

The exact optimization settings, learning-rate schedule, data augmentation, checkpoint-selection procedure, and table-specific reproduction instructions are documented in:

```text
ultralytics/cfg/model/reproducibility.md
```

## Equivalent Ultralytics CLI

The released Ultralytics CLI architecture dispatch accepts the RT-DETR-GMI filename format and normalizes hyphens and underscores before identifying RT-DETR models.

An equivalent CLI-style training command can therefore use:

```bash
yolo detect train \
  model=ultralytics/cfg/model/RT-DETR-GMI.yaml \
  data=path/to/data.yaml \
  epochs=300 \
  patience=40 \
  batch=16 \
  imgsz=640 \
  optimizer=AdamW \
  lr0=0.0001 \
  lrf=1.0 \
  momentum=0.9 \
  weight_decay=0.0001 \
  seed=0 \
  deterministic=True \
  amp=False \
  cache=False \
  workers=4
```

The complete command reproducing the remaining experimental arguments is provided in:

```text
ultralytics/cfg/model/reproducibility.md
```

## Pretrained Initialization

The RT-DETR models used in the reported experiments were instantiated from their YAML architecture configurations.

No external detector checkpoint or pretrained `.pt` weight was explicitly loaded before the reported initial training runs.

Historical Ultralytics training records may contain:

```text
pretrained=True
```

because this was a framework runtime argument. It does not identify an additional explicitly loaded pretrained checkpoint for these experiments.

The reproduction training command therefore does not require an external pretrained checkpoint.

## Validation and Test Evaluation

Validate a trained checkpoint by providing both the checkpoint and dataset YAML paths.

Example:

```bash
python tools/val.py \
  --model runs/train/exp/weights/best.pt \
  --data path/to/data.yaml \
  --split test
```

The fixed test subset must be used for the final reported evaluation:

```text
split = test
```

The training-time validation subset is used for checkpoint selection and the MSLA architecture-selection experiments reported in Tables 4 and 5.

The evaluation protocol, checkpoint-selection procedure, and distinction between validation and test subsets are documented in:

```text
ultralytics/cfg/model/reproducibility.md
```

`tools/val.py` writes a `paper_data.txt` summary under the validation output directory.

## Machine-Readable Results

The repository provides machine-readable numerical results supporting the repeated-run statistics, ablation studies, detector comparisons, per-class analyses, second-dataset evaluation, and representative confusion matrices reported in the revised manuscript.

The principal files are:

```text
reproducibility/results/per_seed_metrics.csv
reproducibility/results/per_class_metrics.csv
reproducibility/results/confusion_matrix_rtdetr_r18_seed0.csv
reproducibility/results/confusion_matrix_rtdetr_gmi_seed0.csv
```

Repeated-run detection metrics are reported as:

```text
mean ± standard deviation
```

using independent runs with:

```text
seed = 0
seed = 42
seed = 3407
```

Params and GFLOPs are deterministic model-level quantities and are therefore reported as single values rather than mean ± standard deviation.

## External Baseline Reproduction

The YOLO and D-FINE baselines were trained using their corresponding framework-specific implementations rather than an artificial unified training script.

Detailed reproduction information is provided in:

```text
reproducibility/external_baselines.md
```

This file records, where available:

```text
framework and package version
repository and release information
source revision
model configuration
dataset configuration
random seeds
initialization procedure
training hyperparameters
data augmentation
framework-specific training command
test-set evaluation command
FPS calculation procedure
```

Where an exact historical upstream commit cannot be recovered from the archived local project, this limitation is stated explicitly instead of assigning an unsupported revision.

The controlled comparison conditions include:

```text
same PVEL-AD train/validation/test split
input resolution = 640 × 640
three independent runs with seeds 0, 42, and 3407
evaluation on the fixed test subset
standardized FPS reporting
```

## Bounding-Box Loss Switching

The original loss-function experiments were conducted by changing the active IoU-based regression-loss line in:

```text
ultralytics/models/utils/loss.py
```

specifically in:

```text
DETRLoss._get_loss_bbox()
```

Detailed source-level instructions are provided in:

```text
reproducibility/loss_switching.md
```

The standard RT-DETR baseline and all ablation configurations that do not include the proposed loss use:

```text
GIoU
```

with Inner-MPDIoU disabled.

The final RT-DETR-GMI configuration and Table 7 configurations containing component `d` use:

```text
Inner-MPDIoU
ratio = 0.7
mpdiou_hw = 2
```

where:

```text
d = Inner-MPDIoU
```

Table 6 compares the following IoU-based regression losses while keeping the architecture and remaining training settings unchanged:

```text
GIoU
Inner-GIoU
Inner-DIoU
Inner-CIoU
Inner-EIoU
MPDIoU
Inner-MPDIoU
```

Only one IoU-based loss should be active at a time.

When reproducing the RT-DETR baseline, Tables 4–5, or Table 7 configurations without component `d`, users must explicitly use the standard GIoU implementation rather than the Inner-MPDIoU line contained in the final RT-DETR-GMI source configuration.

## MSLA Ablation Reproduction

Tables 4 and 5 were implemented as source-level MSLA ablations rather than through additional YAML hyperparameters.

The corresponding source variants are provided under:

```text
ultralytics/nn/extra_modules/
```

Table 4 compares:

```text
1 branch
2 branches
4 branches
```

while keeping the kernel size fixed at `3 × 3`.

Table 5 compares four-branch kernel configurations:

```text
(3, 3, 3, 3)
(1, 3, 5, 7)
(3, 5, 7, 9)
(5, 7, 9, 11)
```

The final selected MSLA configuration is:

```text
4 branches
kernel sizes = (3, 5, 7, 9)
```

The detector YAML remains unchanged during these experiments; only the corresponding MSLA encoder source variant is replaced.

Detailed instructions are provided in:

```text
ultralytics/nn/extra_modules/README.md
ultralytics/cfg/model/reproducibility.md
```

## Parameter Verification

The repository provides:

```text
tools/verify_params.py
```

to verify the parameter-count consistency between the standard AIFI encoder and the MSLA-AIFI encoder used in the manuscript.

The expected counts for:

```text
c1 = 256
cm = 1024
num_heads = 8
```

are:

```text
AIFI:       789,760
MSLA-AIFI:  619,588
Difference: 170,172
```

corresponding to approximately `0.17M` fewer parameters.

## Export

Export a trained checkpoint with:

```bash
python tools/export.py \
  --model runs/train/exp/weights/best.pt \
  --format onnx
```

Exported files such as `*.onnx` are ignored by Git.

## Heatmap Visualization

Generate a Grad-CAM heatmap with a trained checkpoint and an image or image directory:

```bash
python tools/heatmap.py \
  --weight runs/train/exp/weights/best.pt \
  --source path/to/image_or_dir \
  --output runs/heatmap
```

The Grad-CAM helper requires the `grad-cam` package.

## Model Profiling

The repository provides:

```text
main_profile.py
```

for profiling the released RT-DETR-GMI architecture.

Example:

```bash
python main_profile.py \
  --model ultralytics/cfg/model/RT-DETR-GMI.yaml \
  --imgsz 640 640
```

## Reproducibility

The repository provides reproducibility materials supporting the experiments reported in Tables 3–11 of the revised manuscript.

The main reproducibility guide is:

```text
ultralytics/cfg/model/reproducibility.md
```

Additional focused reproduction documents are:

```text
reproducibility/external_baselines.md
reproducibility/loss_switching.md
ultralytics/nn/extra_modules/README.md
```

Together, these documents cover:

```text
experimental environment
training settings
random seeds 0, 42, and 3407
fixed PVEL-AD dataset splits
fixed PV-Multi-Defect dataset splits
pretrained-initialization clarification
validation-based checkpoint selection
MSLA branch-number ablation
MSLA kernel-configuration ablation
source-level IoU-loss switching
sequential component ablation
external YOLO and D-FINE baseline protocols
test-set evaluation
FPS reporting
per-table reproduction instructions
machine-readable per-seed and per-class numerical results
parameter-count verification
```

For experiments involving MSLA, the experimental encoder used to generate the reported results is provided in:

```text
ultralytics/nn/extra_modules/MSLA.py
```

alongside the Table 4/5 branch and kernel variants.

For the external detector comparison, use:

```text
reproducibility/external_baselines.md
```

For reproducing baseline, alternative IoU-loss, and Inner-MPDIoU experiments, use:

```text
reproducibility/loss_switching.md
```

The original image datasets are not redistributed in this repository. Researchers should obtain the corresponding public datasets separately and use the released split manifests to reconstruct the train, validation, and test subsets used in the reported experiments.

## Checkpoints and Model Weights

This repository does not redistribute pretrained or manuscript-trained model weights.

Researchers should train the released configurations using the documented experimental settings or provide their own compatible checkpoint paths when running validation, export, profiling, or visualization tools.

## Third-Party Code and License Notes

This project includes an Ultralytics-derived codebase and third-party components used by RT-DETR-GMI.

Original copyright and license notices are preserved where present.

The root:

```text
LICENSE
```

contains the AGPL-3.0 license text used for the Ultralytics-derived framework code.

Additional third-party attribution and licensing information is provided in:

```text
THIRD_PARTY_NOTICES.md
third_party_licenses/
```

Researchers reusing individual third-party components should review the corresponding upstream licenses and attribution requirements.
