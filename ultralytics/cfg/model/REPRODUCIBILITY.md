# Reproducibility Guide for RT-DETR-GMI

This document records the experimental protocol, model-selection procedure, source-level ablation procedures, dataset splits, and released numerical results used to reproduce the experiments reported in Tables 3–11 of the revised manuscript.

Some ablation experiments were implemented by directly modifying the corresponding source code rather than by introducing additional YAML hyperparameters. To avoid describing a configuration mechanism that was not used in the reported experiments, the source-level modifications required to reproduce these ablations are documented below.

The repository also provides exact train/validation/test split manifests and machine-readable numerical results supporting the reported repeated-run statistics, per-class analyses, ablation studies, cross-model comparisons, second-dataset evaluation, and representative confusion matrices.

---

## 1. Experimental environment

The main experiments were conducted with the following environment:

* GPU: NVIDIA GeForce RTX 4090
* CPU: AMD EPYC 7502 32-Core Processor
* Python: 3.10.14
* PyTorch: 2.2.2
* CUDA: 12.1

The repeated experiments were conducted independently using the following random seeds:

```text
seed = 0
seed = 42
seed = 3407
```

Unless otherwise stated, the remaining training settings were kept unchanged across repeated runs.

---

## 2. Common training protocol

The common RT-DETR training settings are:

| Setting | Value |
| --- | ---: |
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

A representative training configuration is:

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
cache: false
workers: 4
val: true
split: val
iou: 0.7
max_det: 300
lr0: 0.0001
lrf: 1.0
momentum: 0.9
weight_decay: 0.0001
```

For the seed-42 and seed-3407 runs, only the random seed is changed unless otherwise specified.

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

The warm-up duration is 3.0 epochs, with:

```text
warmup_momentum = 0.8
warmup_bias_lr = 0.1
```

---

## 4. Data augmentation

The data-augmentation settings recorded for the experiments are:

| Augmentation parameter | Value |
| --- | ---: |
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

No additional test-time augmentation was enabled:

```yaml
augment: false
```

---

## 5. Dataset splits

### 5.1 PVEL-AD

The PVEL-AD subset used in this study contains 4,481 images from eight defect categories:

* black core
* crack
* finger
* horizontal dislocation
* short circuit
* star crack
* thick line
* vertical dislocation

The fixed split is:

| Subset | Images |
| --- | ---: |
| Train | 3,604 |
| Validation | 395 |
| Test | 482 |

The same split definition is used for all PVEL-AD experiments.

The exact split manifests used in this study are publicly provided under:

```text
reproducibility/splits/
```

Specifically:

```text
reproducibility/splits/pvel_ad_train.txt
reproducibility/splits/pvel_ad_val.txt
reproducibility/splits/pvel_ad_test.txt
```

These files define the exact image membership of the train, validation, and test subsets used in the reported experiments.

Users should prepare the PVEL-AD dataset locally and construct their dataset YAML so that the corresponding directories or image lists match these released manifests.

### 5.2 PV-Multi-Defect

The PV-Multi-Defect dataset version used for the second-dataset experiment contains 1,106 images:

| Subset | Images |
| --- | ---: |
| Train | 771 |
| Validation | 218 |
| Test | 117 |

The exact split manifests are provided under:

```text
reproducibility/splits/pv_multi_defect_train.txt
reproducibility/splits/pv_multi_defect_val.txt
reproducibility/splits/pv_multi_defect_test.txt
```

RT-DETR-r18 and RT-DETR-GMI were independently retrained on the PV-Multi-Defect training subset before evaluation on its fixed test subset.

Therefore, this experiment is a second-dataset reproducibility evaluation under dataset-specific retraining and is not a zero-shot cross-dataset domain-generalization experiment.

---

## 6. Released numerical results

Machine-readable numerical results underlying the statistics reported in the manuscript are provided under:

```text
reproducibility/results/
```

The released files include:

```text
reproducibility/results/per_seed_metrics.csv
reproducibility/results/per_class_metrics.csv
reproducibility/results/confusion_matrix_rtdetr_r18_seed0.csv
reproducibility/results/confusion_matrix_rtdetr_gmi_seed0.csv
```

### `per_seed_metrics.csv`

This file contains the retained run-level numerical results underlying the repeated-run summaries reported for:

* the primary paired RT-DETR-r18 versus RT-DETR-GMI comparison
* MSLA branch-number ablation
* MSLA kernel-configuration ablation
* bounding-box loss comparison
* sequential component ablation
* cross-model comparison
* PV-Multi-Defect evaluation

The file records the dataset, source table, split, experiment group, model or configuration, seed/run identifier, detection metrics, model complexity information where applicable, and FPS measurements where available.

### `per_class_metrics.csv`

This file contains the class-wise numerical results used for the per-class analyses reported in the manuscript.

These include the class-wise Precision, Recall, and AP50 measurements used to calculate the repeated-run summaries and the component-level class-wise analysis.

### Representative confusion matrices

The normalized numerical confusion matrices corresponding to the representative fixed-seed (`seed = 0`) models shown in Fig. 7 are provided as:

```text
confusion_matrix_rtdetr_r18_seed0.csv
confusion_matrix_rtdetr_gmi_seed0.csv
```

The confusion matrices are illustrative fixed-seed analyses and are separate from the three-run quantitative comparisons reported in the main tables.

---

## 7. Reporting convention for repeated experiments

Where repeated experiments are reported, three independent training runs are used:

```text
0
42
3407
```

Detection metrics are summarized as:

```text
mean ± standard deviation
```

using the three independently trained runs.

For the MSLA validation experiments, F1-score is calculated separately for each run from that run's Precision and Recall:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

The mean and sample standard deviation are then calculated across the three run-level F1 values.

Params and GFLOPs are deterministic model-level quantities and are therefore reported as single values rather than mean ± SD.

---

## 8. Model configuration files

The repository provides the relevant RT-DETR model YAML files under:

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

The corresponding released model configurations should be used instead of machine-specific local paths from the original experimental environment.

---

# Table-specific reproduction

## 9. Table 3 — Paired statistical comparison between RT-DETR-r18 and RT-DETR-GMI

Table 3 reports the paired statistical analysis for the primary comparison between:

```text
RT-DETR-r18
RT-DETR-GMI
```

on the PVEL-AD test subset.

No additional training is required for Table 3. The table uses the same independently trained models used in the principal repeated experiments:

```text
seed = 0
seed = 42
seed = 3407
```

For each seed, paired differences are defined as:

```text
RT-DETR-GMI − RT-DETR-r18
```

and are reported in percentage points.

The paired seed-wise values used in the manuscript are:

| Metric | Seed 0 | Seed 42 | Seed 3407 |
| --- | --- | --- | --- |
| mAP50 | 80.32 / 83.98 | 79.33 / 81.69 | 81.26 / 81.85 |
| mAP50:95 | 55.09 / 57.83 | 54.15 / 56.95 | 55.89 / 57.27 |

Seed-wise values are reported as:

```text
RT-DETR-r18 / RT-DETR-GMI
```

The statistical analysis applies a paired two-sided Student's t-test separately to:

```text
mAP50
mAP50:95
```

For each metric, the analysis reports:

```text
mean paired difference
t statistic
degrees of freedom
p value
95% confidence interval
Cohen's dz
```

with:

```text
df = n − 1 = 2
```

and:

```text
Cohen's dz =
mean paired difference /
standard deviation of paired differences
```

The values reported in Table 3 are:

| Metric | Mean difference (pp) | t | df | p | 95% CI (pp) | Cohen's dz |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| mAP50 | 2.20 | 2.477 | 2 | 0.132 | [-1.62, 6.03] | 1.43 |
| mAP50:95 | 2.31 | 4.975 | 2 | 0.038 | [0.31, 4.30] | 2.87 |

Statistical significance is defined in the manuscript as:

```text
p < 0.05
```

Accordingly, the mAP50 comparison does not reach the predefined significance threshold, whereas the mAP50:95 comparison reaches the p < 0.05 threshold.

Because only three paired observations are available, these inferential statistics should be interpreted cautiously and together with the per-seed results and descriptive statistics.

The underlying per-seed values are available in:

```text
reproducibility/results/per_seed_metrics.csv
```

---

## 10. Table 4 — Validation-set ablation study on the number of MSLA branches

Table 4 evaluates the number of MSLA branches under the following architecture:

```text
GCS Backbone: enabled
MSLA-AIFI: enabled
MPMF-Net: disabled
Inner-MPDIoU: disabled
```

The convolutional kernel size is fixed at `3 × 3` for every branch.

The evaluated variants are:

| Branches | Kernel configuration |
| ---: | --- |
| 1 | (3) |
| 2 | (3, 3) |
| 4 | (3, 3, 3, 3) |

The original implementation performed this experiment by modifying the `MSLA` source code directly.

The following equivalent variants reconstruct the branch structures reported in Table 4.

### 10.1 One branch

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

### 10.2 Two branches

The channels are divided equally into two branches, both using `3 × 3` depthwise convolution:

```python
class MSLA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3_1 = DepthwiseConv(
            dim // 2,
            kernel_size=3
        )

        self.dw_conv_3x3_2 = DepthwiseConv(
            dim // 2,
            kernel_size=3
        )

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

### 10.3 Four branches

The channels are divided into four equal branches:

```python
split_size = c // 4
```

All four depthwise convolutions use a `3 × 3` kernel:

```python
self.dw_conv_3x3_1 = DepthwiseConv(
    dim // 4,
    kernel_size=3
)

self.dw_conv_3x3_2 = DepthwiseConv(
    dim // 4,
    kernel_size=3
)

self.dw_conv_3x3_3 = DepthwiseConv(
    dim // 4,
    kernel_size=3
)

self.dw_conv_3x3_4 = DepthwiseConv(
    dim // 4,
    kernel_size=3
)

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

All Table 4 variants were independently trained using:

```text
seed = 0
seed = 42
seed = 3407
```

The configurations were compared on the **validation subset**.

The four-branch configuration was selected based on validation performance and fixed before the subsequent kernel-configuration comparison and corresponding test-set inference.

The run-level validation results underlying Table 4 are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

---

## 11. Table 5 — Validation-set ablation study on the convolutional kernel configuration of MSLA

Table 5 fixes the number of MSLA branches to four and changes only the four depthwise-convolution kernel sizes.

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
self.dw_conv_3x3 = DepthwiseConv(
    dim // 4,
    kernel_size=3
)

self.dw_conv_5x5 = DepthwiseConv(
    dim // 4,
    kernel_size=5
)

self.dw_conv_7x7 = DepthwiseConv(
    dim // 4,
    kernel_size=7
)

self.dw_conv_9x9 = DepthwiseConv(
    dim // 4,
    kernel_size=9
)
```

Only the four kernel sizes are changed between the Table 5 configurations.

The following remain unchanged:

```text
channel partition
linear-attention implementation
branch weighting
concatenation
final 1 × 1 convolution
dataset split
training settings
```

Each configuration was independently trained using:

```text
seed = 0
seed = 42
seed = 3407
```

and compared on the **validation subset**.

The selected MSLA configuration is:

```text
4 branches
kernel sizes = (3, 5, 7, 9)
```

The `(3, 5, 7, 9)` configuration was fixed as the final MSLA-AIFI kernel configuration before the corresponding test-set inference.

The run-level validation results underlying Table 5 are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

---

## 12. Table 6 — Comparison of loss functions

Table 6 compares different IoU-based bounding-box regression losses under the fixed architecture:

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

For each Table 6 experiment:

```text
Only one IoU-based loss is enabled.
All other architecture and training settings remain unchanged.
```

The RT-DETR-GMI methodology uses:

```text
Inner-MPDIoU
ratio = 0.7
mpdiou_hw = 2
```

The implementation uses image-normalized bounding-box coordinates.

Therefore:

```text
W_n = 1
H_n = 1
```

and the corner-distance normalization denominator is:

```text
D = W_n^2 + H_n^2
  = 1^2 + 1^2
  = 2
```

This corresponds directly to:

```python
mpdiou_hw=2
```

in the released implementation.

The Inner-IoU scaling ratio used in all reported Inner-MPDIoU experiments is:

```python
ratio=0.7
```

Table 6 is a **post-selection test-set comparison** used to characterize the empirical behavior of the predefined Inner-MPDIoU loss relative to alternative regression losses.

The Table 6 test results were not used to select the loss function or to perform subsequent architecture or hyperparameter tuning.

The underlying run-level results are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

---

## 13. Table 7 — Ablation experiments on the components of RT-DETR-GMI

Table 7 uses:

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

The corresponding released model configurations should be used wherever available.

Each configuration was independently trained with:

```text
seed = 0
seed = 42
seed = 3407
```

using the same PVEL-AD split and common training protocol.

Table 7 was conducted after the model design had been fixed.

Its test-set results were used to characterize the individual and combined contributions of the proposed components and were not used to guide subsequent architecture or hyperparameter selection.

The run-level numerical results are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

---

## 14. Table 8 — Performance comparison of different detectors on PVEL-AD

Table 8 compares RT-DETR-GMI with the general-purpose detector baselines reported in the manuscript.

For the RT-DETR family, the repository provides:

```text
rtdetr-r18.yaml
rtdetr-r34.yaml
rtdetr-r50.yaml
```

All compared models were evaluated using the same PVEL-AD dataset split and the standardized evaluation protocol used in the manuscript.

The quantitative comparison is based on **three independent runs**.

The repeated-run protocol uses:

```text
seed = 0
seed = 42
seed = 3407
```

Precision, Recall, mAP50, mAP50:95, and FPS are reported as:

```text
mean ± SD
```

over the three runs.

Params and GFLOPs are fixed model-level values.

The external YOLO and D-FINE baselines were trained using their corresponding official implementations available at the time of the experiments.

Because these models originate from different repositories and software versions, their configuration formats, command-line interfaces, training entry points, and software defaults are not identical.

A single artificial unified training command would therefore not accurately represent the actual external implementations used in the experiments.

For this reason, the repository documents the common controlled experimental conditions:

```text
same PVEL-AD dataset split
input resolution = 640 × 640
three independent runs
standardized test-set evaluation
standardized FPS reporting
```

rather than retrospectively rewriting all external implementations into one command format.

Where an external baseline is reproduced independently, researchers should use the corresponding official implementation and model configuration associated with that model rather than assuming that the latest software defaults are identical to those used in this study.

The complete retained run-level numerical records underlying Table 8 are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

The Table 8 comparison is descriptive. Differences between the reported means are not interpreted as establishing statistical superiority across all compared detector families.

---

## 15. Table 9 — Per-class Precision, Recall, and AP50 comparison

Table 9 reports class-wise:

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

The class-wise values are obtained from the same independently trained models used for the principal repeated comparison.

The runs use:

```text
seed = 0
seed = 42
seed = 3407
```

and are summarized as mean ± SD.

No separate Table 9 training experiment is required.

The underlying class-wise numerical results are provided in:

```text
reproducibility/results/per_class_metrics.csv
```

---

## 16. Table 10 — Incremental class-wise AP50 changes in the sequential RT-DETR-GMI ablation

Table 10 is derived from the sequential component-ablation models used in Table 7.

The model sequence is:

```text
RT-DETR-r18
→ GCS
→ GCS + MSLA
→ GCS + MSLA + MPMF
→ GCS + MSLA + MPMF + Inner-MPDIoU
```

The corresponding models are the same independently trained configurations used for the Table 7 sequential ablation.

No independent Table 10 training experiment is required.

For each defect category, Table 10 reports the change in three-run mean AP50 between adjacent configurations in the sequential ablation pathway.

The underlying class-wise records are provided in:

```text
reproducibility/results/per_class_metrics.csv
```

---

## 17. Table 11 — Performance comparison on PV-Multi-Defect

Table 11 evaluates:

```text
RT-DETR-r18
RT-DETR-GMI
```

on PV-Multi-Defect.

Both models are trained directly on the PV-Multi-Defect training subset before evaluation.

The fixed dataset split is:

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

The exact dataset membership is provided by:

```text
reproducibility/splits/pv_multi_defect_train.txt
reproducibility/splits/pv_multi_defect_val.txt
reproducibility/splits/pv_multi_defect_test.txt
```

The run-level numerical results are provided in:

```text
reproducibility/results/per_seed_metrics.csv
```

This experiment evaluates reproducibility under a second photovoltaic-dataset setting with dataset-specific retraining.

It is not a zero-shot or direct cross-dataset domain-transfer experiment.

---

## 18. Model-selection and test-set evaluation protocol

The training, validation, and test subsets have different roles.

### Training subset

The training subset is used for parameter optimization.

### Validation subset

The validation subset is used during training for checkpoint selection.

The framework evaluates the validation subset using:

```yaml
val: true
split: val
```

The training framework computes detection fitness as:

```text
fitness = 0.1 × mAP50 + 0.9 × mAP50:95
```

with zero weights assigned to Precision and Recall in the fitness calculation.

The best validation checkpoint produced by the training framework is used for subsequent evaluation.

The validation subset was also used for the MSLA architecture-selection experiments reported in Tables 4 and 5.

The selection sequence was:

```text
1. Compare 1, 2, and 4 MSLA branches on validation data.
2. Fix the selected branch number at 4.
3. Compare the candidate four-branch kernel configurations on validation data.
4. Fix (3, 5, 7, 9) as the final MSLA kernel configuration.
```

### Test subset

The test subset was not used to select:

* the MSLA branch number
* the MSLA kernel configuration
* training hyperparameters
* training checkpoint

Inner-MPDIoU was defined as part of the proposed RT-DETR-GMI methodology before the post-selection loss comparison reported in Table 6.

Table 6 evaluates fixed loss alternatives after the model-design stage in order to characterize their empirical behavior.

Table 7 similarly provides post-selection component-ablation results after the model design had been fixed.

Accordingly, multiple fixed configurations were evaluated on the test subset for post-selection analysis.

The procedure should therefore not be interpreted as a single physical inference pass over the test set.

However, the reported test-set results were not used to revise model-selection decisions or to perform subsequent architecture or hyperparameter tuning.

No architecture-selection decision was changed after examining the post-selection test-set results.

---

## 19. Equivalent RT-DETR training command

The experiments were conducted through the Ultralytics RT-DETR training pipeline.

The following command reconstructs the common RT-DETR training settings.

Replace:

```text
<MODEL_CONFIG>
<DATA_CONFIG>
```

with the corresponding released model configuration and local dataset YAML.

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

For repeated experiments, rerun the same configuration with:

```text
seed=0
seed=42
seed=3407
```

Hardware-specific device assignment may be changed when reproducing the experiments on another workstation.

The model architecture, dataset split, optimization settings, and random seed should otherwise follow the protocol documented above.

---

## 20. Final test evaluation

After the required validation-based model selection and checkpoint selection are complete, the selected checkpoint is evaluated on the fixed test split.

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

The principal inference settings reported in the manuscript are:

```text
input resolution = 640 × 640
batch size = 16
precision = FP32
backend = native PyTorch
```

FPS represents end-to-end inference throughput under the standardized evaluation setting, including preprocessing, model inference, and postprocessing.

For Tables 4 and 5, architecture selection is based on validation-set performance rather than test-set performance.

For Tables 6 and 7, the reported test-set experiments are post-selection analyses and were not used to guide subsequent model tuning.

---

## 21. Key training arguments

The key training values used in the RT-DETR experiments are summarized below:

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

The corresponding repeated runs change the random seed to:

```text
42
3407
```

while keeping the remaining experimental conditions unchanged unless otherwise documented.

---

## 22. Important note on source-level ablations

Tables 4–6 were performed as source-level ablations rather than through dedicated YAML switches.

Specifically:

* **Table 4:** modifies the number of MSLA branches.
* **Table 5:** modifies only the four MSLA depthwise-convolution kernel sizes.
* **Table 6:** modifies only the active IoU-based bounding-box regression loss.

This document intentionally records those source-level changes instead of introducing a new configuration mechanism that was not used in the reported experiments.

---

## 23. Reproducibility checklist

To reproduce a reported experiment:

1. Prepare the corresponding public dataset.
2. Use the exact released train/validation/test split manifest.
3. Select the corresponding released model YAML.
4. Apply the documented source-level modification when reproducing Tables 4–6.
5. Keep all remaining training arguments unchanged.
6. Train independently with seeds `0`, `42`, and `3407` where repeated runs are reported.
7. Use the validation subset for MSLA architecture selection and checkpoint selection.
8. Select the best validation checkpoint generated by the training framework.
9. Evaluate the fixed checkpoint on the corresponding test subset.
10. For Tables 6 and 7, treat the test-set results as post-selection characterization rather than model-selection evidence.
11. Do not use post-selection test results to revise architecture or training hyperparameters.
12. Report mean ± SD for repeated-run metrics and fixed values for Params and GFLOPs.
13. Use the released machine-readable numerical results for comparison with the manuscript summaries.

The exact dataset split manifests are available under:

```text
reproducibility/splits/
```

and the numerical results underlying the manuscript summaries are available under:

```text
reproducibility/results/
```
