# Ultralytics YOLO 🚀, AGPL-3.0 license
"""RT-DETR-GMI public module registry.

Only modules required by ``RT-DETR-GMI.yaml`` are re-exported here. This avoids
registering unrelated experimental/vendored modules in the minimal public release.
"""

from .gmi import C2f_MambaOut, MFM
from .msla_reimplementation import TransformerEncoderLayer_MSLA

# ``parse_model`` contains an upstream-compatible optional branch for an ``Index``
# module even though RT-DETR-GMI.yaml does not use it.  Defining the name keeps
# that optional check from raising NameError while preserving current behavior.
Index = None

__all__ = 'C2f_MambaOut', 'MFM', 'TransformerEncoderLayer_MSLA', 'Index'
