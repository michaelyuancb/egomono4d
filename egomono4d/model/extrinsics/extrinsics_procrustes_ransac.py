from dataclasses import dataclass
from typing import Literal
import numpy as np

import torch
from jaxtyping import Float
from torch import Tensor

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from ..projection import align_surfaces_eval
from .extrinsics import Extrinsics


@dataclass
class ExtrinsicsProcrustesRANSACCfg:
    name: Literal["procrustes_ransac"]
    max_iter: int | None
    num_points: int | None


class ExtrinsicsProcrustesRANSAC(Extrinsics[ExtrinsicsProcrustesRANSACCfg]):
    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        surfaces: Float[Tensor, "batch frame height width 3"],
    ) -> Float[Tensor, "batch frame 4 4"]:
        device = surfaces.device
        _, _, h, w, _ = surfaces.shape

        indices = torch.linspace(0, h*w-1, self.cfg.num_points, dtype=torch.int64, device=device,)
        best_extrinsics, best_score = align_surfaces_eval(surfaces, flows.backward, backbone_output.weights, batch.flys, indices) 
        for i in range(self.cfg.max_iter):
            maybe_inliers = np.random.choice(h*w, size=self.cfg.num_points, replace=False)
            extrinsics, score =  align_surfaces_eval(surfaces, flows.backward, backbone_output.weights, batch.flys, maybe_inliers) 
            if score > best_score:
                # print(f"undate score: {score} > {best_score}")
                best_score = score 
                best_extrinsics = extrinsics 
        
        return extrinsics
