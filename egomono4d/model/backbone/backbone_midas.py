from dataclasses import dataclass
from typing import Literal
import pdb

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ...tracking.track_predictor import Tracks
from ..projection import earlier, later, sample_image_grid
from .backbone import Backbone, BackboneOutput


def make_net(dims):
    def init_weights_normal(m):
        if type(m) == nn.Linear:
            if hasattr(m, "weight"):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
                )

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    net = nn.Sequential(*layers[:-1])
    net.apply(init_weights_normal)
    return net


@dataclass
class BackboneMidasCfg:
    name: Literal["midas"]
    pretrained: bool
    weight_sensitivity: float | None
    mapping: Literal["original", "exp"]
    model: Literal["DPT_Large", "MiDaS_small"]
    local_dir: str | None


class BackboneMidas(Backbone[BackboneMidasCfg]):
    def __init__(
        self,
        cfg: BackboneMidasCfg,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
        patch_size: tuple[int, int] | None,
    ) -> None:
        super().__init__(cfg, num_frames=num_frames, image_shape=image_shape, patch_size=patch_size)
        print("start loading midas")
        if cfg.local_dir is None:
            self.midas = torch.hub.load(
                "intel-isl/MiDaS",
                cfg.model,
                pretrained=cfg.pretrained,
            )
        else:
            print(f"load midas from {cfg.local_dir}")
            self.midas = torch.hub.load(
                cfg.local_dir,
                cfg.model,
                pretrained=cfg.pretrained,
                trust_repo=True, source='local'
            )
        print("finish loading midas")
        self.midas_out = self.midas.scratch.output_conv
        self.midas.scratch.output_conv = nn.Identity()

        # If a weight sensitivity is specified, don't learn weights.
        if cfg.weight_sensitivity is None:
            weight_channels = {
                "DPT_Large": 256,
                "MiDaS_small": 64,
            }[cfg.model]
            self.corr_weighter_perpoint = make_net([weight_channels*2, 128, 64, 1])
            self.weight_channels = weight_channels
            # self.corr_weighter_perpoint = make_net([weight_channels * 2 + 2, 128, 64, 1])
        else:
            weights = torch.full((num_frames - 1, *image_shape), 0, dtype=torch.float32)
            self.weights = nn.Parameter(weights)

        if cfg.mapping == "exp":
            self.midas_out = nn.Sequential(*self.midas_out[:-2])


        self.intr_encoder = nn.Sequential(
            nn.Conv3d(in_channels=2*self.weight_channels, out_channels=128, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=(1, 0, 0)),  # (b, 512, f, h, w) --> (b, 128, f, h/5, w/5)
            nn.Tanh(),
            nn.AvgPool3d((1, 3, 3)),                     # (b, 128, f, h/15, w/15) --> (b, 32, f, h/15, w/15)
        )
        self.focal_decoder = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 0, 0)),     # (b, 128, f, h/15, w/15) --> (b, 64, f, h/45, w/45)
            nn.AdaptiveAvgPool3d((1, 1, 1)),    # (b, 32, 1, 1, 1)
            nn.Tanh()
        )
        self.focal_predictor = nn.Sequential( 
            nn.Linear(32, 2),
            nn.Softplus()
        )
        self.principle_decoder = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 0, 0)),     # (b, 128, f, h/15, w/15) --> (b, 64, f, h/45, w/45)
            nn.AdaptiveAvgPool3d((1, 1, 1)),    # (b, 32, 1, 1, 1)
            nn.Tanh()
        )
        self.principle_predictor = nn.Sequential( 
            nn.Linear(32, 2),
            nn.Sigmoid()
        )


    def forward(self, batch: Batch, flows: Flows | list[Tracks]) -> BackboneOutput:
        device = batch.videos.device
        b, f, _, h, w = batch.videos.shape

        videos = rearrange(batch.videos, "b f c h w -> (b f) c h w")
        features = self.midas(videos)
        psedo_disparity = self.midas_out(features)

        # This matches Cameron's original implementation.
        match self.cfg.mapping:
            case "original":
                depths = 1e3 / (psedo_disparity + 0.1)
            case "exp":
                depths = (psedo_disparity / 1000).exp() + 0.01

        features = F.interpolate(features, (h, w), mode="bilinear") / 20

        depths = rearrange(depths, "(b f) () h w -> b f h w", b=b, f=f)
        features = rearrange(features, "(b f) c h w -> b f c h w", b=b, f=f)

        # Compute correspondence weights.
        if self.cfg.weight_sensitivity is None:
            xy, _ = sample_image_grid((h, w), device)
            backward_weights, intrinsics = self.compute_correspondence_weights(
                self.grid_sample_features(earlier(features), xy + flows.backward),
                later(features),
            )

        else:
            backward_weights = (self.cfg.weight_sensitivity * self.weights).sigmoid()
            backward_weights = backward_weights[None]

        return BackboneOutput(depths, backward_weights, intrinsics)

    def compute_correspondence_weights(
        self,
        features_earlier: Float[Tensor, "batch frame channel height width"],
        features_later: Float[Tensor, "batch frame channel height width"],
        # masks_earlier: Float[Tensor, "batch frame channel height width"],
        # masks_later: Float[Tensor, "batch frame channel height width"]
    ) -> Float[Tensor, "batch frame height width"]:
        # features = torch.cat((features_earlier, features_later, masks_earlier, masks_later), dim=2)
        features = torch.cat((features_earlier, features_later), dim=2)
    
        w_features = rearrange(features, "b f c h w -> b f h w c")
        weights = self.corr_weighter_perpoint(w_features).sigmoid().clip(min=0)

        i_features = rearrange(features, "b f c h w -> b c f h w")
        i_features = self.intr_encoder(i_features)
        b, f = features.shape[0], features.shape[1]
        focals_features = self.focal_decoder(i_features)
        principles_features = self.principle_decoder(i_features)
        focals = self.focal_predictor(focals_features.reshape(b, -1))
        principles = self.principle_predictor(principles_features.reshape(b, -1))

        return rearrange(weights, "b f h w () -> b f h w"), (focals, principles)

    def grid_sample_features(
        self,
        features: Float[Tensor, "batch frame channel height width"],
        grid: Float[Tensor, "batch frame height width xy=2"],
    ) -> Float[Tensor, "batch frame channel height width"]:
        b, f, _, _, _ = features.shape
        samples = F.grid_sample(
            rearrange(features, "b f c h w -> (b f) c h w"),
            rearrange(grid * 2 - 1, "b f h w xy -> (b f) h w xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rearrange(samples, "(b f) c h w -> b f c h w", b=b, f=f)
