# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL, 2022
from mmedit.models import MODELS as MMEdit_MODELS

# from mmcv.utils import Registry

# MODELS = Registry('model', parent=MMEdit_MODELS)
MODELS = MMEdit_MODELS
BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS
