# Reproducibility Guide for RT-DETR-GMI

This document records the experimental protocol and source-level ablation procedures used to reproduce the experiments reported in Tables 3–10 of the manuscript.

Some ablation experiments were implemented by directly modifying the corresponding source code rather than by introducing additional YAML hyperparameters. To avoid describing a configuration mechanism that was not used in the experiments, the exact source-level changes required to reproduce those ablations are documented below.

---

## 1. Experimental environment

The main experiments were conducted with the following environment:

- GPU: NVIDIA GeForce RTX 4090
- CPU: AMD EPYC 7502 32-Core Processor
- Python: 3.10.14
- PyTorch: 2.2.2
- CUDA: 12.1

The representative training record supplied with this repository corresponds to a seed-0 run. For repeated experiments, the seed was changed to `42` and `3407` while the remaining training settings were kept unchanged.

---

## 2. Common training protocol

The common RT-DETR training settings are:

| Setting | Value |
|---|---:|
| Epochs | 300 |
| Early-stopping patience | 40 |
| Batch size | 16 |
| Input resolution | 640 × 640 |
| Optimizer | AdamW |
| Initial learning rate (`lr0`) | 0.0001 |
| Final LR factor (`lrf`) | 1.0 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Deterministic training | True |
| Pretrained initialization | True |
| AMP | False |
| Cache | False |
| Workers | 4 |
| Nominal batch size (`nbs`) | 64 |
| Validation during training | True |
| Training-time validation split | `val` |
| Validation IoU | 0.7 |
| Maximum detections | 300 |

The repeated experiments use:

```text
seed = 0
seed = 42
seed = 3407
```

The supplied training record for the representative seed-0 run contains:

```yaml
epochs: 300
patience: 40
batch: 16
imgsz: 640
pretrained: true
optimizer: AdamW
seed: 0
deterministic: true
cos_lr: false
amp: false
val: true
split: val
iou: 0.7
max_det: 300
lr0: 0.0001
lrf: 1.0
momentum: 0.9
weight_decay: 0.0001
```

---

## 3. Learning-rate schedule

The training configuration uses:

```yaml
cos_lr: false
lr0: 0.0001
lrf: 1.0
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
```

The intended warm-up duration is 3.0 epochs. The remaining warm-up settings are `warmup_momentum=0.8` and `warmup_bias_lr=0.1`.

---

## 4. Data augmentation

The data-augmentation settings recorded for the experiments are:

| Augmentation parameter | Value |
|---|---:|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `degrees` | 0.0 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `shear` | 0.0 |
| `perspective` | 0.0 |
| `flipud` | 0.0 |
| `fliplr` | 0.5 |
| `mosaic` | 0.0 |
| `mixup` | 0.0 |
| `copy_paste` | 0.0 |
| `close_mosaic` | 0 |

No additional test-time augmentation was enabled (`augment: false`).

---

## 5. Dataset splits

### 5.1 PVEL-AD

The PVEL-AD subset used in this study contains 4,481 images from eight defect categories:

- black core
- crack
- finger
- horizontal dislocation
- short circuit
- star crack
- thick line
- vertical dislocation

The fixed split is:

| Subset | Images |
|---|---:|
| Train | 3,604 |
| Validation | 395 |
| Test | 482 |

The same split definition is used for all PVEL-AD experiments.

The representative local training record used:

```text
/home/lxt/DETR-0914/split_4481/data.yaml
```

This path is machine-local and is shown only to document the original training setup. The repository does not distribute the original train/validation/test split manifest files. Users should prepare the dataset according to the split sizes reported above and point their local dataset YAML to the corresponding train, validation, and test directories.

### 5.2 PV-Multi-Defect

The dataset version used for the second-dataset experiment contains 1,106 images:

| Subset | Images |
|---|---:|
| Train | 771 |
| Validation | 218 |
| Test | 117 |

RT-DETR-r18 and RT-DETR-GMI were separately retrained on the PV-Multi-Defect training subset before evaluation.

---

## 6. Model configuration files

The repository provides the relevant model YAML files under:

```text
ultralytics/cfg/model/
```

including:

```text
rtdetr-r18.yaml
rtdetr-r34.yaml
rtdetr-r50.yaml
rtdetr-GCS.yaml
rtdetr-AIFI-MSLA.yaml
rtdetr-MPMF-Net.yaml
```

The final RT-DETR-GMI architecture combines:

```text
GCS Backbone
+ MSLA-AIFI
+ MPMF-Net
+ Inner-MPDIoU
```

The original representative training record referred to the local model path:

```text
/home/lxt/DETR-0914/12.yaml
```

This path is retained here only as provenance for the original run; repository users should use the corresponding released model configuration.

---

# Table-specific reproduction

## 7. Table 3 — MSLA branch-number ablation

Table 3 evaluates the number of MSLA branches under the following architecture:

```text
GCS Backbone: enabled
MSLA-AIFI: enabled
MPMF-Net: disabled
Inner-MPDIoU: disabled
```

The convolutional kernel size is fixed at `3 × 3` for every branch.

The evaluated variants are:

| Branches | Kernel configuration |
|---:|---|
| 1 | (3) |
| 2 | (3, 3) |
| 4 | (3, 3, 3, 3) |

The original implementation performed this experiment by modifying the `MSLA` source code directly. The following equivalent source variants reconstruct the branch structures reported in Table 3.

### 7.1 One branch

The full channel dimension is processed by one branch:

```python
class MSLA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim, kernel_size=3)
        self.linear_attention = LinearAttention(
            dim=dim,
            num_heads=num_heads
        )
        self.final_conv = nn.Conv2d(dim, dim, 1)
        self.scale_weights = nn.Parameter(
            torch.ones(1),
            requires_grad=True
        )

    def forward(self, input_):
        b, n, c = input_.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)

        input_reshaped = input_.reshape([b, c, h, w])

        x = self.dw_conv_3x3(input_reshaped)
        att = self.linear_attention(x)

        processed_input = att * self.scale_weights[0]
        final_output = self.final_conv(processed_input)

        return final_output.reshape(b, n, self.dim)
```

### 7.2 Two branches

The channels are divided equally into two branches, both using `3 × 3` depthwise convolution:

```python
class MSLA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3_1 = DepthwiseConv(dim // 2, kernel_size=3)
        self.dw_conv_3x3_2 = DepthwiseConv(dim // 2, kernel_size=3)

        self.linear_attention = LinearAttention(
            dim=dim // 2,
            num_heads=num_heads
        )

        self.final_conv = nn.Conv2d(dim, dim, 1)
        self.scale_weights = nn.Parameter(
            torch.ones(2),
            requires_grad=True
        )

    def forward(self, input_):
        b, n, c = input_.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)

        input_reshaped = input_.reshape([b, c, h, w])
        split_size = c // 2

        x1 = input_reshaped[:, :split_size, :, :]
        x2 = input_reshaped[:, split_size:, :, :]

        x1 = self.dw_conv_3x3_1(x1)
        x2 = self.dw_conv_3x3_2(x2)

        att1 = self.linear_attention(x1)
        att2 = self.linear_attention(x2)

        processed_input = torch.cat([
            att1 * self.scale_weights[0],
            att2 * self.scale_weights[1]
        ], dim=1)

        final_output = self.final_conv(processed_input)
        return final_output.reshape(b, n, self.dim)
```

### 7.3 Four branches

The channels are divided into four equal branches:

```python
split_size = c // 4
```

All four depthwise convolutions use a `3 × 3` kernel:

```python
self.dw_conv_3x3_1 = DepthwiseConv(dim // 4, kernel_size=3)
self.dw_conv_3x3_2 = DepthwiseConv(dim // 4, kernel_size=3)
self.dw_conv_3x3_3 = DepthwiseConv(dim // 4, kernel_size=3)
self.dw_conv_3x3_4 = DepthwiseConv(dim // 4, kernel_size=3)

self.linear_attention = LinearAttention(
    dim=dim // 4,
    num_heads=num_heads
)

self.scale_weights = nn.Parameter(
    torch.ones(4),
    requires_grad=True
)
```

The four outputs are concatenated along the channel dimension before the final `1 × 1` convolution.

All Table 3 variants are trained with the same training protocol and random seeds `0`, `42`, and `3407`.

---

## 8. Table 4 — MSLA kernel-configuration ablation

Table 4 fixes the number of MSLA branches to four and changes only the four depthwise-convolution kernel sizes.

The architecture remains:

```text
GCS Backbone: enabled
MSLA-AIFI: enabled
MPMF-Net: disabled
Inner-MPDIoU: disabled
```

The evaluated kernel configurations are:

```text
(3, 3, 3, 3)
(1, 3, 5, 7)
(3, 5, 7, 9)
(5, 7, 9, 11)
```

For example, the selected final configuration is implemented as:

```python
self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)
```

Only the four kernel sizes are changed between Table 4 runs. The channel partition, linear-attention implementation, branch weighting, concatenation, final `1 × 1` convolution, dataset split, and training settings remain unchanged.

The selected MSLA configuration is:

```text
4 branches
kernel sizes = (3, 5, 7, 9)
```

---

## 9. Table 5 — Bounding-box loss comparison

Table 5 compares different IoU-based bounding-box regression losses under the fixed architecture:

```text
GCS Backbone
+ MSLA-AIFI
+ MPMF-Net
```

The L1 bounding-box term remains unchanged:

```python
loss[name_bbox] = (
    self.loss_gain['bbox']
    * F.l1_loss(
        pred_bboxes,
        gt_bboxes,
        reduction='sum'
    )
    / len(gt_bboxes)
)
```

Only the active IoU-based loss line in `_get_loss_bbox()` is changed.

### GIoU

```python
loss[name_giou] = 1.0 - bbox_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    GIoU=True
)
```

### Inner-GIoU

```python
loss[name_giou] = 1.0 - bbox_inner_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    GIoU=True,
    ratio=0.7
)
```

### Inner-DIoU

```python
loss[name_giou] = 1.0 - bbox_inner_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    DIoU=True,
    ratio=0.7
)
```

### Inner-CIoU

```python
loss[name_giou] = 1.0 - bbox_inner_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    CIoU=True,
    ratio=0.7
)
```

### Inner-EIoU

```python
loss[name_giou] = 1.0 - bbox_inner_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    EIoU=True,
    ratio=0.7
)
```

### MPDIoU

```python
loss[name_giou] = 1.0 - bbox_mpdiou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    mpdiou_hw=2
)
```

### Inner-MPDIoU

```python
loss[name_giou] = 1.0 - bbox_inner_mpdiou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    mpdiou_hw=2,
    ratio=0.7
)
```

For each Table 5 experiment:

```text
Only one IoU-loss line is enabled.
All other architecture and training settings remain unchanged.
```

The final RT-DETR-GMI model uses:

```text
Inner-MPDIoU
ratio = 0.7
mpdiou_hw = 2
```

The implementation uses normalized bounding-box coordinates. Therefore the image-coordinate normalization corresponds to:

```text
W_n = 1
H_n = 1
W_n^2 + H_n^2 = 2
```

which is consistent with the fixed `mpdiou_hw=2` used in the released implementation.

---

## 10. Table 6 — Component ablation

Table 6 uses:

```text
a = GCS Backbone
b = MSLA-AIFI
c = MPMF-Net
d = Inner-MPDIoU
```

The evaluated configurations are:

```text
RT-DETR-r18 baseline

a
b
c
d

a + b
a + b + c
a + b + c + d
```

The corresponding released model configuration files should be used wherever available.

Each configuration is independently trained with:

```text
seed = 0
seed = 42
seed = 3407
```

using the same PVEL-AD split and common training protocol.

---

## 11. Table 7 — Comparison with other detectors

Table 7 compares RT-DETR-GMI with the general-purpose detector baselines reported in the manuscript.

For the RT-DETR family, the repository provides:

```text
rtdetr-r18.yaml
rtdetr-r34.yaml
rtdetr-r50.yaml
```

All models retrained in this study use the same PVEL-AD split, input resolution, and evaluation protocol stated in the manuscript.

The external YOLO and D-FINE baselines were trained using their corresponding official implementations available at the time of the experiments. Because these baselines belong to different repositories and software versions, their configuration formats, command-line interfaces, default options, and training entry points are not identical. Therefore, a single unified command cannot accurately represent all Table 7 baselines, and retroactively rewriting them into one common command format could introduce settings that were not actually used.

For this reason, this repository documents the common controlled conditions used for the comparison rather than reconstructing model-specific commands from later software versions. The controlled conditions include:

```text
PVEL-AD dataset split
input resolution = 640 × 640
fixed random seed = 0 for the Table 7 single-run comparison
test-set evaluation under the standardized protocol reported in the manuscript
```

Where a baseline is reproduced from an external repository, users should use the corresponding official implementation and model configuration from that repository version rather than assuming that current default commands are identical to those used in the study.

This limitation does not affect the reported Table 7 results, but it means that exact command syntax for every external baseline is not distributed in this repository.

---

## 12. Table 8 — Per-class comparison

Table 8 reports class-wise:

```text
Precision
Recall
AP50
```

for:

```text
RT-DETR-r18
RT-DETR-GMI
```

The class-wise values are obtained from the same independently trained models used for the main comparison and are summarized over seeds:

```text
0
42
3407
```

as mean ± SD.

No separate Table 8 training run is required.

---

## 13. Table 9 — Class-wise component analysis

Table 9 is derived from the sequential component-ablation models used in Table 6.

The model sequence is:

```text
RT-DETR-r18
→ GCS
→ GCS + MSLA
→ GCS + MSLA + MPMF
→ GCS + MSLA + MPMF + Inner-MPDIoU
```

The same checkpoints produced for the corresponding Table 6 configurations are used for the class-wise AP50 analysis.

No independent Table 9 training procedure is required.

---

## 14. Table 10 — PV-Multi-Defect

Table 10 evaluates:

```text
RT-DETR-r18
RT-DETR-GMI
```

on PV-Multi-Defect.

Both models are trained directly on the PV-Multi-Defect training subset before evaluation.

Dataset split:

```text
Train:      771
Validation: 218
Test:       117
```

Each model is independently trained with:

```text
seed = 0
seed = 42
seed = 3407
```

using the same principal optimization protocol as the PVEL-AD experiments.

This experiment is a second-dataset reproducibility evaluation and is not a zero-shot or cross-dataset domain-generalization test.

---

## 15. Model selection and checkpoint selection

Architecture and hyperparameter selection are performed using the validation subset.

The test subset is reserved for final evaluation after the architecture and training protocol have been fixed.

During training:

```yaml
val: true
split: val
```

The framework evaluates the validation set during training. The released implementation computes detection fitness as:

```text
fitness = 0.1 × mAP50 + 0.9 × mAP50:95
```

with zero weights for Precision and Recall in the fitness calculation.

The best validation checkpoint produced by the training framework is used for final test evaluation.

The test subset is not used to select:

- MSLA branch number
- MSLA kernel configuration
- bounding-box loss
- model architecture
- training hyperparameters
- checkpoint

---

## 16. Equivalent training command

The original experiment was run through the Ultralytics RT-DETR training pipeline. The following command is an equivalent reconstruction of the supplied `args.yaml` settings.

Replace `<MODEL_CONFIG>` and `<DATA_CONFIG>` with the released repository paths.

```bash
yolo detect train \
  model=<MODEL_CONFIG> \
  data=<DATA_CONFIG> \
  epochs=300 \
  patience=40 \
  batch=16 \
  imgsz=640 \
  optimizer=AdamW \
  lr0=0.0001 \
  lrf=1.0 \
  momentum=0.9 \
  weight_decay=0.0001 \
  warmup_epochs=3.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  seed=0 \
  deterministic=True \
  pretrained=True \
  amp=False \
  cache=False \
  workers=4 \
  cos_lr=False \
  close_mosaic=0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  degrees=0.0 \
  translate=0.1 \
  scale=0.5 \
  shear=0.0 \
  perspective=0.0 \
  flipud=0.0 \
  fliplr=0.5 \
  mosaic=0.0 \
  mixup=0.0 \
  copy_paste=0.0
```

For the repeated experiments, rerun the same configuration with:

```text
seed=0
seed=42
seed=3407
```

The supplied seed-0 record used two visible CUDA devices:

```text
device=0,1
```

Users reproducing the experiments on different hardware may adjust only the device assignment while keeping the optimization and model settings unchanged.

---

## 17. Final test evaluation

After model selection is completed using the validation subset, evaluate the selected best checkpoint on the fixed test split.

An equivalent Ultralytics evaluation command is:

```bash
yolo detect val \
  model=<BEST_CHECKPOINT> \
  data=<DATA_CONFIG> \
  split=test \
  imgsz=640 \
  batch=16 \
  iou=0.7 \
  max_det=300 \
  half=False \
  plots=True
```

The manuscript reports FP32 evaluation using the native PyTorch backend.

The test results are used only for final reporting, not for architecture, hyperparameter, loss-function, or checkpoint selection.

---

## 18. Raw training arguments

For transparency, the key training values used for reproduction are listed below. The warm-up duration is corrected to the intended value of `3.0` epochs:

```yaml
task: detect
mode: train
epochs: 300
patience: 40
batch: 16
imgsz: 640
save: true
save_period: -1
cache: false
device: 0,1
workers: 4
pretrained: true
optimizer: AdamW
seed: 0
deterministic: true
rect: false
cos_lr: false
close_mosaic: 0
resume: false
amp: false
fraction: 1.0
dropout: 0.0
val: true
split: val
conf: null
iou: 0.7
max_det: 300
half: false
augment: false
agnostic_nms: false
lr0: 0.0001
lrf: 1.0
momentum: 0.9
weight_decay: 0.0001
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
label_smoothing: 0.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
```

---

## 19. Important note on source-level ablations

Tables 3–5 were performed as source-level ablations rather than through dedicated YAML switches:

- **Table 3:** modifies the number of MSLA branches.
- **Table 4:** modifies only the four MSLA depthwise-convolution kernel sizes.
- **Table 5:** modifies only the active IoU-based bounding-box regression loss.

This document intentionally records those source-level changes instead of introducing a new configuration mechanism that was not used in the reported experiments.

---

## 20. Reproducibility checklist

To reproduce a reported experiment:

1. Use the fixed dataset split.
2. Select the corresponding released model YAML.
3. Apply the documented source-level modification if reproducing Tables 3–5.
4. Keep all other training arguments unchanged.
5. Train independently with seeds `0`, `42`, and `3407` where repeated runs are reported.
6. Perform architecture/hyperparameter selection using the validation subset only.
7. Select the best validation checkpoint.
8. Evaluate the selected checkpoint on the fixed test subset.
9. Report the corresponding single-run result or mean ± SD according to the table protocol.
10. Do not use the test subset for model selection.


