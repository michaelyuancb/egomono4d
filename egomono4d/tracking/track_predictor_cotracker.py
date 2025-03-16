from dataclasses import dataclass
# from typing import Literal
from typing_extensions import Literal
import pdb
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .track_predictor import TrackPredictor, Tracks, sample_image_grid_tracker
import cotracker
from cotracker.predictor import CoTrackerPredictor


@dataclass
class TrackPredictorCoTrackerCfg:
    name: Literal["cotracker"]
    grid_size: int
    similarity_threshold: float
    cache_dir: str | None
    # cache_path: str | None


class TrackPredictorCoTracker(TrackPredictor[TrackPredictorCoTrackerCfg]):
    def __init__(self, cfg: TrackPredictorCoTrackerCfg) -> None:
        super().__init__(cfg)
        self.cache_dir = cfg.cache_dir
        # checkpoint = "scaled_offline.pth"
        checkpoint = "cotracker2.pth"
        self.tracker = CoTrackerPredictor(checkpoint=cfg.cache_dir+"/cotracker_checkpoints/"+checkpoint)
        grid_size = self.cfg.grid_size
        self.grid_queries = sample_image_grid_tracker((grid_size, grid_size))[None]
        self.grid_queries_init = False
        
        
    def calc_tracking(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
        query_frame: int,
        backward_tracking: bool=True
    ) -> Tracks:

        # (Michael) Ensuring that the coordinates of tracking points is INT for loss_tracking_robust.
        b, _, _, h, w = videos.shape
        if self.grid_queries_init is False:
            gs = self.grid_queries.clone()
            gs[..., 0] = gs[..., 0] * (w - 1) 
            gs[..., 1] = gs[..., 1] * (h - 1)
            gs = torch.round(gs).to(videos.device)
            self.grid_queries = gs.reshape(1, -1, 2)
            self.grid_queries_init = True

        queries = torch.cat([torch.zeros_like(self.grid_queries[:, :, :1], device=videos.device) * query_frame, self.grid_queries], dim=-1)

        # pdb.set_trace()
        xy, visibility = self.tracker(videos*255, queries=queries.repeat(b, 1, 1), grid_query_frame=query_frame, backward_tracking=backward_tracking)
        xy, visibility = self.tracker(
            videos * 255,
            queries=queries.repeat(b, 1, 1),
            # grid_size=self.cfg.grid_size,
            grid_query_frame=query_frame,
            backward_tracking=backward_tracking,
        )

        # Normalize the coordinates.
        b, f, _, h, w = videos.shape
        wh = torch.tensor((w-1, h-1), dtype=torch.float32, device=videos.device)
        xy = xy / wh

        # Filter visibility based on RGB values.
        rgb = F.grid_sample(
            rearrange(videos, "b f c h w -> (b f) c h w"),
            rearrange(xy, "b f p xy -> (b f) p () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        rgb = rearrange(rgb, "(b f) c p () -> b f p c", b=b, f=f)
        rgb_delta = (rgb[:, [query_frame]] - rgb).abs().norm(dim=-1)
        visibility = visibility & (rgb_delta < self.cfg.similarity_threshold)

        return Tracks(xy, visibility, 0)

    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
        query_frame: int,
    ) -> Tracks:
    
        if query_frame > 1:
            return self.calc_tracking(videos, query_frame, backward_tracking=True)
        elif query_frame == 0:
            return self.calc_tracking(videos, query_frame, backward_tracking=False)
        else:
            raise ValueError(f"Unsupport query_frame for co-trackerr, query_frame={query_frame}")

