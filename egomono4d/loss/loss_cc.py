from dataclasses import dataclass
from typing import Literal
import torch
import pdb

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..tracking import Tracks
from .loss import Loss, LossCfgCommon

@dataclass
class LossCCCfg(LossCfgCommon):    # CC: Clip Consistency
    name: Literal["cc"]


class LossCC(Loss[LossCCCfg]):
    def __init__(self, cfg: LossCCCfg) -> None:
        super().__init__(cfg)
        self.loss = torch.nn.L1Loss(reduction="none")

    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        current_epoch: int,
        return_val: bool
    ) -> tuple[Float[Tensor, ""], dict]:
        
        intrinsics = model_output.intrinsics
        b, f, _, _ = intrinsics.shape
        assert b % 2 == 0
        rb = b // 2

        intrinsics_subclip_1 = intrinsics[::2]     # (b//2, f, 3, 3)
        intrinsics_subclip_2 = intrinsics[1::2]    # (b//2, f, 3, 3)
        loss = self.loss(intrinsics_subclip_1, intrinsics_subclip_2)
        loss = loss.sum() / (rb * f)                 # fx, fy, cx, cy 

        return loss, {"cc": loss}