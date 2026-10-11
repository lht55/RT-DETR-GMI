# RT-DETR Bounding-Box Loss Switching

This document describes how the IoU-based bounding-box regression loss was switched in the RT-DETR experiments.

The original experiments used source-level loss switching in:

```text
ultralytics/models/utils/loss.py
```

specifically in the `_get_loss_bbox()` function of `DETRLoss`.

The L1 bounding-box regression term was kept unchanged in all experiments:

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

Only the IoU-based regression-loss line assigned to `loss[name_giou]` was changed.

## Switching rule

Only one IoU-based loss line should be active at a time.

All other candidate IoU-loss lines must remain commented out.

For example, when reproducing the standard RT-DETR baseline or an ablation that does not include Inner-MPDIoU, the active loss must be the standard GIoU implementation:

```python
loss[name_giou] = 1.0 - bbox_iou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    GIoU=True
)
```

and the Inner-MPDIoU line must be disabled:

```python
# loss[name_giou] = 1.0 - bbox_inner_mpdiou(
#     pred_bboxes,
#     gt_bboxes,
#     xywh=True,
#     mpdiou_hw=2,
#     ratio=0.7
# )
```

For the final RT-DETR-GMI model, the standard GIoU line is disabled and Inner-MPDIoU is enabled:

```python
# loss[name_giou] = 1.0 - bbox_iou(
#     pred_bboxes,
#     gt_bboxes,
#     xywh=True,
#     GIoU=True
# )

loss[name_giou] = 1.0 - bbox_inner_mpdiou(
    pred_bboxes,
    gt_bboxes,
    xywh=True,
    mpdiou_hw=2,
    ratio=0.7
)
```

## Loss settings used in the experiments

### Standard GIoU

Used for the RT-DETR baseline and for component ablations that do not include the proposed Inner-MPDIoU loss.

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

## Mapping to the reported experiments

For the standard RT-DETR baselines:

```text
Active loss: GIoU
Inner-MPDIoU: disabled
```

For Tables 4 and 5, which evaluate MSLA architecture variants:

```text
Active loss: GIoU
Inner-MPDIoU: disabled
```

For the component ablations in Table 7 that do not include component `d`:

```text
Active loss: GIoU
Inner-MPDIoU: disabled
```

where:

```text
d = Inner-MPDIoU
```

For the final configuration containing component `d`:

```text
Active loss: Inner-MPDIoU
ratio = 0.7
mpdiou_hw = 2
```

For Table 6, the architecture and all other training settings remain unchanged, and only the active IoU-based loss line in `_get_loss_bbox()` is replaced by the corresponding loss listed above.

## Important reproducibility note

The released source currently contains the Inner-MPDIoU implementation used by the final RT-DETR-GMI model.

When reproducing the RT-DETR baseline or any ablation without Inner-MPDIoU, users must explicitly restore the standard GIoU line and disable the Inner-MPDIoU line before training.

This source-level switching procedure reflects how the reported loss-function and component-ablation experiments were originally conducted and prevents Inner-MPDIoU from being unintentionally applied to baseline or non-loss ablation configurations.
