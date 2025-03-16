from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor
import pdb
import torch

from ...dataset.types import Batch, BatchInference
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from .intrinsics import Intrinsics
from ...tracking.track_predictor import Tracks


@dataclass
class IntrinsicsModelCfg:
    name: Literal["model"]


class IntrinsicsModel(Intrinsics[IntrinsicsModelCfg]):
    def forward(
        self,
        batch: Batch | BatchInference,
        flows: Flows | list[Tracks],
        backbone_output: BackboneOutput,
        global_step: int,
    ) -> Float[Tensor, "batch frame 3 3"]:
        # Just return the ground-truth intrinsics.
        # pdb.set_trace()
        b, f, _, h, w = batch.videos.shape
        focal, principle = backbone_output.intrinsics
        focal = focal * (h * w) ** 0.5
        intrinsics = torch.stack([torch.eye(3, dtype=torch.float32, device=focal.device)]*b, dim=0)
        intrinsics = torch.stack([intrinsics]*f, dim=1)
        intrinsics[..., :2, 2] = principle.unsqueeze(-2)
        intrinsics[..., 0, 0] = focal[..., 0].unsqueeze(-1) / w
        intrinsics[..., 1, 1] = focal[..., 1].unsqueeze(-1) / h
        return intrinsics