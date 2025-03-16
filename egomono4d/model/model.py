from dataclasses import dataclass
import pdb
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch, BatchInference
from ..flow.flow_predictor import Flows
from ..tracking.track_predictor import Tracks
from .backbone import BackboneCfg, get_backbone
from .extrinsics import ExtrinsicsCfg, get_extrinsics
from .intrinsics import IntrinsicsCfg, get_intrinsics
from .projection import sample_image_grid, unproject


@dataclass
class ModelCfg:
    backbone: BackboneCfg
    intrinsics: IntrinsicsCfg
    extrinsics: ExtrinsicsCfg
    use_correspondence_weights: bool


@dataclass
class ModelOutput:
    depths: Float[Tensor, "batch frame height width"] 
    surfaces: Float[Tensor, "batch frame height width xyz=3"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    extrinsics: Float[Tensor, "batch frame 4 4"]
    backward_correspondence_weights: Float[Tensor, "batch frame-1 height width"]


@dataclass
class ModelExports:
    extrinsics: Float[Tensor, "batch frame 4 4"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    colors: Float[Tensor, "batch frame 3 height width"]
    depths: Float[Tensor, "batch frame height width"]


class Model(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
        num_frames: int | None = None,
        image_shape: tuple[int, int] | None = None,
        patch_size: tuple[int, int] | None = None,
        inference_trunc: float | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = get_backbone(cfg.backbone, num_frames, image_shape, patch_size)
        self.intrinsics = get_intrinsics(cfg.intrinsics)
        self.extrinsics = get_extrinsics(cfg.extrinsics, num_frames)
        self.inference_trunc = inference_trunc

    def forward(
        self,
        batch: Batch | BatchInference,
        flows: Flows | list[Tracks],
        global_step: int,
        masks_hoi_aux: Tensor | None = None,
    ) -> ModelOutput:
        device = batch.videos.device
        _, f, _, h, w = batch.videos.shape   # (B, T, C, H, W)

        # Run the backbone, which provides depths and correspondence weights.
        backbone_out = self.backbone.forward(batch, flows)

        # Allow the correspondence weights to be ignored as an ablation.
        if not self.cfg.use_correspondence_weights:
            backbone_out.weights = torch.ones_like(backbone_out.weights)
        else:
            if self.inference_trunc is not None:
                backbone_out.weights[backbone_out.weights < self.inference_trunc] = 0.0
        
        # pdb.set_trace()
        if masks_hoi_aux is not None:
            backbone_out.weights[masks_hoi_aux[:, 1:] > 0] = 0.0

        # backbone_out.weights[:, :, -60:] = 0.0

        # Compute the intrinsics.
        intrinsics = self.intrinsics.forward(batch, flows, backbone_out, global_step)

        # Use the intrinsics to calculate camera-space surfaces (point clouds).
        xy, _ = sample_image_grid((h, w), device=device)
        # import pdb 
        # pdb.set_trace()
        surfaces = unproject(
            xy,
            backbone_out.depths,
            rearrange(intrinsics, "b f i j -> b f () () i j"),
        )

        # Finally, compute the extrinsics.
        extrinsics = self.extrinsics.forward(batch, flows, backbone_out, surfaces)

        return ModelOutput(
            backbone_out.depths,
            surfaces,
            intrinsics,
            extrinsics,
            backbone_out.weights,
        )

    @torch.no_grad()
    def export(
        self,
        batch: Batch,
        flows: Flows,
        global_step: int,
    ) -> ModelExports:
        # For now, only implement exporting with a batch size of 1.
        b, _, _, _, _ = batch.videos.shape
        assert b == 1

        output = self.forward(batch, flows, global_step)

        return ModelExports(
            output.extrinsics,
            output.intrinsics,
            batch.videos,
            output.depths,
        )
