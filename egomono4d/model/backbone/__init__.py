from .backbone import Backbone
from .backbone_explicit_depth import BackboneExplicitDepth, BackboneExplicitDepthCfg
from .backbone_midas import BackboneMidas, BackboneMidasCfg
from .backbone_unidepth import BackboneUniDepth, BackboneUniDepthCfg
from .backbone_nvds_unidepth import BackboneNvdsUniDepth, BackboneNvdsUniDepthCfg

try:
    from .backbone_depthanythingv2 import BackboneDepthanythingV2, BackboneDepthanythingV2Cfg
    from .backbone_nvds_unet_dpt import BackboneNvdsUnetDPT, BackboneNvdsUnetDPTCfg
except:
    BackboneDepthanythingV2 = None
    BackboneDepthanythingV2Cfg = None 
    BackboneNvdsUnetDPT = None
    BackboneNvdsUnetDPTCfg = None

BACKBONES = {
    "explicit_depth": BackboneExplicitDepth,
    "midas": BackboneMidas,
    "unidepth": BackboneUniDepth,
    "depthanythingv2": BackboneDepthanythingV2,
    "nvds_unet_dpt": BackboneNvdsUnetDPT,
    "nvds_unidepth": BackboneNvdsUniDepth
}

BackboneCfg = BackboneExplicitDepthCfg | BackboneMidasCfg | BackboneNvdsUniDepthCfg | \
    BackboneNvdsUnetDPTCfg | BackboneDepthanythingV2Cfg | BackboneUniDepthCfg


def get_backbone(
    cfg: BackboneCfg,
    num_frames: int | None,
    image_shape: tuple[int, int] | None,
    patch_size: tuple[int, int] | None = None,
) -> Backbone:
    return BACKBONES[cfg.name](cfg, num_frames, image_shape, patch_size)
