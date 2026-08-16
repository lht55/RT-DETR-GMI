# Third-Party Notices

This minimal public release is scoped to the source needed by `RT-DETR-GMI.yaml`.
Unrelated experimental modules and vendored CUDA/C++ operators are excluded from the
prepared Git set via `.gitignore`.

Keep all existing `LICENSE`, `AUTHORS`, `NOTICE`, and source-level copyright/license
headers intact.

## Bundled Third-Party Source

| Project / module | Current repository path | Original upstream repository | Version / commit basis | Original license | Modified in this project |
| --- | --- | --- | --- | --- | --- |
| Ultralytics YOLO / RT-DETR codebase | `ultralytics/`, `setup.py`, `setup.cfg` | https://github.com/ultralytics/ultralytics | Local package version is `8.0.201` (`ultralytics/__init__.py`) | AGPL-3.0; root `LICENSE` contains AGPL-3.0 text | Yes: project entry points, RT-DETR-GMI module registration, and model configuration were modified for this release |
| MambaOut | `ultralytics/nn/backbone/MambaOut.py` | https://github.com/yuweihao/MambaOut | exact upstream revision unknown; current upstream `main` checked at `9f2f2343eb0f99f2cf3ba6b92290b5a81be2bad1` is similar but not an exact match to the local modified file | Apache-2.0; local copy retained at `third_party_licenses/MambaOut-APACHE-2.0.txt` | Modified: import paths, YOLO/RT-DETR feature-output adaptation, `GatedCNNBlock_BCHW`, and local metadata |
| DCMPNet MFM | `ultralytics/nn/extra_modules/gmi.py` (`MFM`) | https://github.com/zhoushen1/DCMPNet | Adapted from `models/DIACMPN.py`; current upstream `main` checked at `e21e2a0adea78de57753d7397c628a51035426bd` | MIT; local copy retained at `third_party_licenses/DCMPNet-MIT.txt` | Modified: adapted constructor to accept Ultralytics channel lists and added per-input `1x1` projection |

## Project-Specific RT-DETR-GMI Modules

| Module | Current repository path | Notes |
| --- | --- | --- |
| GMI module registry / `C2f_MambaOut` wrapper | `ultralytics/nn/extra_modules/__init__.py`, `ultralytics/nn/extra_modules/gmi.py` | Local wrapper around the Apache-2.0 MambaOut block. Treat wrapper code as project code only if confirmed by the project owner. |
| Independent MSLA encoder | `ultralytics/nn/extra_modules/msla_reimplementation.py` | Clean-room implementation written for this repository from the public MSLAU-Net paper's algorithm description, formulas, and structure diagrams (paper citation retained in README; arXiv:2505.18823). The final public repository does not include the MSLAU-Net GitHub source code or the previous implementation derived from that source. README citation is retained for the paper/algorithmic idea. |

Project owner must confirm that any project-specific portions are original or otherwise
authorized for release under this repository's AGPL-3.0 licensing terms before staging.

## Excluded From This Minimal Public Release

The following local third-party or experimental components are not required by
`RT-DETR-GMI.yaml` after narrowing the module registry and are excluded from the
prepared Git set:

- `ultralytics/models/fastsam/`
- `ultralytics/models/nas/`
- `ultralytics/models/sam/`
- `ultralytics/models/yolo/classify/`
- `ultralytics/models/yolo/model.py`
- `ultralytics/models/yolo/pose/`
- `ultralytics/models/yolo/segment/`
- `ultralytics/cfg/models/`
- `ultralytics/nn/backbone/convnextv2.py`
- `ultralytics/nn/backbone/CSwimTramsformer.py`
- `ultralytics/nn/backbone/EfficientFormerV2.py`
- `ultralytics/nn/backbone/efficientViT.py`
- `ultralytics/nn/backbone/fasternet.py`
- `ultralytics/nn/backbone/faster_cfg/`
- `ultralytics/nn/backbone/lsknet.py`
- `ultralytics/nn/backbone/lsnet.py`
- `ultralytics/nn/backbone/mobilenetv4.py`
- `ultralytics/nn/backbone/overlock.py`
- `ultralytics/nn/backbone/pkinet.py`
- `ultralytics/nn/backbone/repvit.py`
- `ultralytics/nn/backbone/rmt.py`
- `ultralytics/nn/backbone/starnet.py`
- `ultralytics/nn/backbone/SwinTransformer.py`
- `ultralytics/nn/backbone/TransNeXt/`
- `ultralytics/nn/backbone/TransNext.py`
- `ultralytics/nn/backbone/UniRepLKNet.py`
- `ultralytics/nn/backbone/VanillaNet.py`
- `ultralytics/nn/extra_modules/block.py`
- `ultralytics/nn/extra_modules/DCMPNet.py`
- `ultralytics/nn/extra_modules/cutlass/`
- `ultralytics/nn/extra_modules/DCNv4_op/`
- `ultralytics/nn/extra_modules/GroupMamba/`
- `ultralytics/nn/extra_modules/kan_convs/`
- `ultralytics/nn/extra_modules/mamba/`
- `ultralytics/nn/extra_modules/mobileMamba/`
- `ultralytics/nn/extra_modules/ops_dcnv3/`
- `ultralytics/nn/extra_modules/ops_dscn/`
- `ultralytics/nn/extra_modules/rational_kat_cu/`
- `ultralytics/nn/extra_modules/selective_scan/`

These excluded components are not listed as redistributed third-party source for the
minimal public release. If a future release re-includes any of them, repeat the upstream
source, exact version, and license audit for that component first.
