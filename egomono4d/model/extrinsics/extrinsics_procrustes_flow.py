from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from ..projection import align_surfaces
from .extrinsics import Extrinsics


@dataclass
class ExtrinsicsProcrustesFlowCfg:
    name: Literal["procrustes_flow"]
    num_points: int | None
    randomize_points: bool


class ExtrinsicsProcrustesFlow(Extrinsics[ExtrinsicsProcrustesFlowCfg]):
    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        surfaces: Float[Tensor, "batch frame height width 3"],
    ) -> Float[Tensor, "batch frame 4 4"]:
        device = surfaces.device
        _, _, h, w, _ = surfaces.shape

        # Select the subset of points used for the alignment.
        if self.cfg.num_points is None:
            indices = torch.arange(h * w, dtype=torch.int64, device=device)
        elif self.cfg.randomize_points:
            indices = torch.randint(
                0,
                h * w,
                (self.cfg.num_points,),
                dtype=torch.int64,
                device=device,
            )
        else:
            indices = torch.linspace(
                0,
                h * w - 1,
                self.cfg.num_points,
                dtype=torch.int64,
                device=device,
            )

        # Align the depth maps using a Procrustes fit.
        return align_surfaces(
            surfaces,                 # (B, F, H, W, 3)
            flows.backward,           # (B, F-1, H, W, 2)
            backbone_output.weights,  # (B, F-1, H, W)
            indices,                  # rand-index (H*W)
        ) 
        # (B, F, 4, 4)