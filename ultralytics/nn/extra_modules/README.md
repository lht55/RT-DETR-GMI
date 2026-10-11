# MSLA-AIFI Reproducibility

## 1. Source

MSLA-AIFI integrates the MSLA mechanism from **MSLAU-Net** into the RT-DETR AIFI encoder.

Reference: **Lan et al., MSLAU-Net**  
Repository: https://github.com/Monsoon49/MSLAU-Net  
Commit: `efdb058ebe363434376c8ef6cbf689391b60b7fc`

The MSLA mechanism is adopted from MSLAU-Net, while
`TransformerEncoderLayer_MSLA` is the RT-DETR encoder integration used in this study.

---

## 2. Experimental implementation

The implementation used to generate the manuscript results is provided at:

`ultralytics/nn/extra_modules/MSLA.py`

This file was extracted from the original transformer source used during our
experiments. The original training source contained multiple candidate attention
modules evaluated during model development. For reproducibility, unrelated
experimental modules and unused imports were removed, while the MSLA-AIFI
implementation, parameter definitions, and forward computation used in the
reported experiments were preserved.

The final MSLA configuration contains four depthwise-convolution branches with
kernel sizes `(3, 5, 7, 9)`, followed by a shared `LinearAttention` module,
learnable branch scale weights, feature concatenation, and a final `1×1`
convolution.

---

## 3. Ablation protocol

All MSLA-AIFI ablation experiments in Tables 4 and 5 used the **same detector
YAML configuration**.

During the original experiments, the detector architecture and YAML configuration
were kept unchanged. Only the source implementation of
`TransformerEncoderLayer_MSLA` was replaced to evaluate different branch numbers
or depthwise-convolution kernel configurations.

The corresponding source implementations are now provided in
`ultralytics/nn/extra_modules/`.

Therefore, separate detector YAML files are not required for each MSLA ablation;
the source implementation is the only component changed between these experiments.

---

## 4. Table 4: Number of MSLA branches

In Table 4, only the number of MSLA branches was changed. The depthwise
convolution kernel size was fixed at `3×3` for all branches.

| Branches | Kernels | Source implementation | Detector Params |
| ---: | --- | --- | ---: |
| 1 | `(3,)` | `table4_branch1.py` | 13.86M |
| 2 | `(3, 3)` | `table4_branch2.py` | 13.67M |
| 4 | `(3, 3, 3, 3)` | `table5_k3333.py` | 13.62M |

The four-branch `(3,3,3,3)` configuration in Table 4 is identical to the
`(3,3,3,3)` configuration evaluated in Table 5; therefore the same source
implementation is used.

All other components of MSLA-AIFI and the detector were kept unchanged.

---

## 5. Table 5: Depthwise-convolution kernel sizes

In Table 5, the number of branches was fixed at four. Only the depthwise
convolution kernel sizes were changed.

| Kernels | Source implementation | Detector Params |
| --- | --- | ---: |
| `(3, 3, 3, 3)` | `table5_k3333.py` | 13.62M |
| `(1, 3, 5, 7)` | `table5_k1357.py` | 13.62M |
| `(3, 5, 7, 9)` | `table5_k3579.py` | 13.62M |
| `(5, 7, 9, 11)` | `table5_k57911.py` | 13.63M |

The `(3,5,7,9)` configuration was selected as the final MSLA configuration used
in RT-DETR-GMI.

---

## 6. Parameter consistency

For the encoder dimensions used in the experiments
(`c1=256`, `cm=1024`, `num_heads=8`):

| Encoder | Parameters |
| --- | ---: |
| Original AIFI | 789,760 |
| Experimental MSLA-AIFI | 619,588 |
| Reduction | 170,172 |

Thus, replacing AIFI with the experimental MSLA-AIFI reduces the encoder
parameter count by:

`789,760 - 619,588 = 170,172 ≈ 0.17M`

This is consistent with the model parameter change reported in Table 7:

`19.88M -> 19.71M`

The previously released `msla_reimplementation.py` was not the implementation
used to generate the manuscript results. The experimental implementation
provided here is the version corresponding to the reported results.

---

## 7. File mapping

The released MSLA reproducibility files are:

```text
ultralytics/nn/extra_modules/

MSLA.py
table4_branch1.py
table4_branch2.py
table5_k3333.py
table5_k1357.py
table5_k3579.py
table5_k57911.py