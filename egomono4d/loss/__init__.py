from .loss import Loss
from .loss_dynamic_area import LossDynamicArea, LossDynamicAreaCfg
from .loss_shape import LossShape, LossShapeCfg #, loss_shape_func
from .loss_flow_3d import LossFlow3D, LossFlow3DCfg #, loss_flow_3d_func
from .loss_tracking_3d import LossTracking3D, LossTracking3DCfg #, loss_tracking_3d_func
from .loss_cc import LossCC, LossCCCfg

LOSSES = {
    "dynamic_area": LossDynamicArea,
    "tracking_3d": LossTracking3D,
    "flow_3d": LossFlow3D,
    "shape": LossShape,
    "cc": LossCC,
}

LossCfg = LossDynamicAreaCfg | LossTracking3DCfg | LossCCCfg | LossShapeCfg | LossFlow3DCfg


def get_losses(cfgs: list[LossCfg]) -> list[Loss]:
    return [LOSSES[cfg.name](cfg) for cfg in cfgs]
