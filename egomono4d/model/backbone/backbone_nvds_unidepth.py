from dataclasses import dataclass
from typing import Literal
import json
import os
import pdb
import warnings
from math import ceil

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..projection import earlier, later, sample_image_grid
from .backbone import Backbone, BackboneOutput

from .modules.unet3d import UNet3D
from .modules.transformer import Transformer
from unidepth.models import UniDepthV2

IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)
RESOLUTION_LEVELS = 10


# preprocess helpers
def _check_ratio(image_ratio, ratio_bounds):
    ratio_bounds = sorted(ratio_bounds)
    if ratio_bounds is not None and (
        image_ratio < ratio_bounds[0] or image_ratio > ratio_bounds[1]
    ):
        warnings.warn(
            f"Input image ratio ({image_ratio:.3f}) is out of training "
            f"distribution: {ratio_bounds}. This may lead to unexpected results. "
            f"Consider resizing/padding the image to match the training distribution."
        )


def _check_resolution(shape_constraints, resolution_level):
    if resolution_level is None:
        warnings.warn(
            "Resolution level is not set. Using max resolution. "
            "You can tradeoff resolution for speed by setting a number in [0,10]. "
            "This can be achieved by setting model's `resolution_level` attribute."
        )
        resolution_level = RESOLUTION_LEVELS
    pixel_bounds = sorted(shape_constraints["pixels_bounds_ori"])
    pixel_range = pixel_bounds[-1] - pixel_bounds[0]
    clipped_resolution_level = min(max(resolution_level, 0), RESOLUTION_LEVELS)
    if clipped_resolution_level != resolution_level:
        warnings.warn(
            f"Resolution level {resolution_level} is out of bounds ([0,{RESOLUTION_LEVELS}]). "
            f"Clipping to {clipped_resolution_level}."
        )
    shape_constraints["pixels_bounds"] = [
        pixel_bounds[0]
        + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
        pixel_bounds[0]
        + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
    ]
    return shape_constraints


def _get_closes_num_pixels(image_shape, pixels_bounds):
    h, w = image_shape
    num_pixels = h * w
    pixels_bounds = sorted(pixels_bounds)
    num_pixels = max(min(num_pixels, pixels_bounds[1]), pixels_bounds[0])
    return num_pixels


def _shapes(image_shape, shape_constraints):
    h, w = image_shape
    image_ratio = w / h
    _check_ratio(image_ratio, shape_constraints["ratio_bounds"])
    num_pixels = _get_closes_num_pixels(
        (h / shape_constraints["patch_size"], w / shape_constraints["patch_size"]),
        shape_constraints["pixels_bounds"],
    )
    h = ceil((num_pixels / image_ratio) ** 0.5 - 0.5)
    w = ceil(h * image_ratio - 0.5)
    ratio = h / image_shape[0] * shape_constraints["patch_size"]
    return (
        h * shape_constraints["patch_size"],
        w * shape_constraints["patch_size"],
    ), ratio


def _preprocess(rgbs, intrinsics, shapes, ratio):
    rgbs = F.interpolate(rgbs, size=shapes, mode="bilinear", antialias=True)
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(outs, ratio, original_shapes, mode="nearest-exact"):
    outs["depth"] = F.interpolate(outs["depth"], size=original_shapes, mode=mode)
    outs["confidence"] = F.interpolate(
        outs["confidence"], size=original_shapes, mode="bilinear", antialias=True
    )
    outs["K"] = outs["K"].clone()
    outs["K"][:, 0, 0] = outs["K"][:, 0, 0] / ratio
    outs["K"][:, 1, 1] = outs["K"][:, 1, 1] / ratio
    outs["K"][:, 0, 2] = outs["K"][:, 0, 2] / ratio
    outs["K"][:, 1, 2] = outs["K"][:, 1, 2] / ratio
    return outs


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
class BackboneNvdsUniDepthCfg:
    name: Literal["nvds_unidepth"] 
    cache_dir: str | None
    estimator: str                     # unidepth_v2_[large, small]  
    finetune_head: bool               # whether to only finetune dpt head of depth-anything-v2
    
    unet_num: int
    unet_channels: list
    unet_kernel_size: int
    unet_groups: int

    transformer_depth: int
    transformer_heads: int 
    transformer_dim_head: int 
    transformer_mlp_dim: int


class BackboneNvdsUniDepth(Backbone[BackboneNvdsUniDepthCfg]):
    def __init__(
        self,
        cfg: BackboneNvdsUniDepthCfg,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
        patch_size: tuple[int, int] | None,
    ) -> None:
        super().__init__(cfg, num_frames=num_frames, image_shape=image_shape, patch_size=patch_size)

        version = 'v2'
        estimator = cfg.estimator
        if estimator.endswith('large'):
            backbone = 'vitl'
        elif estimator.endswith('small'):
            backbone = 'vits'
        with open(os.path.join(cfg.cache_dir, "unidepth_v2_checkpoints", f"unidepth-{version}-{backbone}14.json"), "r") as f:
            config = json.load(f)
        unidepth = UniDepthV2(config)
        model_dir = os.path.join(cfg.cache_dir, "unidepth_v2_checkpoints", f"unidepth-{version}-{backbone}14.bin")
        unidepth.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.unidepth = unidepth.eval()

        for param in self.unidepth.parameters():
            param.requires_grad = True
        if self.cfg.finetune_head is True:
            for param in self.unidepth.pixel_encoder.parameters():
                param.requires_grad = False          

        # TODO: For Higher-Resolution, Please Set this larger, e.g. [1200, 2400] as in UniDepthV2
        self.unidepth.shape_constraints["pixels_bounds"] = [900, 1200]
        self.unidepth.shape_constraints["pixels_bounds_ori"] = [900, 1200]

        self.embed_dim = self.unidepth.pixel_encoder.embed_dim
        self.weight_channels = config["model"]["pixel_decoder"]["hidden_dim"]
        self.corr_weighter_perpoint = make_net([self.weight_channels*4+2, 512, 64, 1])

        self.unet_num = cfg.unet_num
        self.unet_groups = cfg.unet_groups
        self.unet_channels = cfg.unet_channels
        self.unet_kernel_size = cfg.unet_kernel_size
        self.feature_fusion_unet_list = nn.ModuleList([])
        for i in range(4):
            unet_seq = nn.Sequential()
            for j in range(self.unet_num):
                unet = UNet3D(input_channels=self.embed_dim, feature_channels=self.unet_channels, kernel_size=self.unet_kernel_size, groups=self.unet_groups)
                unet_seq.append(unet)
            self.feature_fusion_unet_list.append(unet_seq)

        self.dino_projector = nn.Sequential(
            nn.Linear(self.embed_dim*4, self.weight_channels*4),
            nn.ReLU(),
            nn.Linear(self.weight_channels*4, self.weight_channels)
        )

        self.token_projector_list = nn.ModuleList([
            Transformer(dim=self.embed_dim, depth=cfg.transformer_depth, heads=cfg.transformer_heads,
                        dim_head=cfg.transformer_dim_head, mlp_dim=cfg.transformer_mlp_dim)
        for i in range(4)])
        self.global_token_projector_list = nn.ModuleList([
            Transformer(dim=self.embed_dim, depth=cfg.transformer_depth, heads=cfg.transformer_heads,
                        dim_head=cfg.transformer_dim_head, mlp_dim=cfg.transformer_mlp_dim)
        for i in range(2)])
        self.camera_token_projector_list = nn.ModuleList([
            Transformer(dim=self.embed_dim, depth=cfg.transformer_depth, heads=cfg.transformer_heads,
                        dim_head=cfg.transformer_dim_head, mlp_dim=cfg.transformer_mlp_dim)
        for i in range(4)])
        

    def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
        
        device = batch.videos.device
        b, f, _, H, W = batch.videos.shape
        rgbs = rearrange(batch.videos, "b f c h w -> (b f) c h w")

        if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
            rgbs = TF.normalize(
                rgbs,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )
        else:
            raise ValueError(f"Please set the pixel value in [0, 1] (with preprocess). [min:{rgbs.min()}] [max:{rgbs.max()}]")

        shape_constraints = _check_resolution(self.unidepth.shape_constraints, self.unidepth.resolution_level)
        (h, w), ratio = _shapes((H, W), shape_constraints) 
        if not ((h==H) and (w==W) and (ratio==1.0)):
            rgbs, gt_intrinsics = _preprocess(rgbs, None, (h, w), ratio)    

        features, tokens = self.unidepth.pixel_encoder(rgbs)  

        cls_tokens = [x.contiguous() for x in tokens]        
        features = [
            self.unidepth.stacking_fn(features[i:j]).contiguous()
            for i, j in self.unidepth.slices_encoder_range           
        ]
        # features:  [torch.Size([20, 42, 56, 1024])] * 4
        tokens = [
            self.unidepth.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.unidepth.slices_encoder_range
        ]
        # tokens:  [torch.Size([20, 1, 1024])] * 4
        global_tokens = [cls_tokens[i] for i in [-2, -1]]                    
        camera_tokens = [cls_tokens[i] for i in [-3, -2, -1]] + [tokens[-2]] 

        _, patch_h, patch_w, _ = features[0].shape


        dino_features_fused = []
        shallow_dino_features = []
        for i, feature in enumerate(features):
            feature = feature.reshape(b, f, patch_h, patch_w, self.embed_dim).permute(0, 4, 1, 2, 3)   # (b, c, f, h, w)
            dino_feat_fused = self.feature_fusion_unet_list[i](feature)
            dino_feat_fused = dino_feat_fused.permute(0, 2, 3, 4, 1)           # (b, f, h, w, c)
            shallow_dino_features.append(dino_feat_fused) 
            dino_features_fused.append(dino_feat_fused.reshape(b*f, patch_h, patch_w, self.embed_dim))

        shallow_dino_features = torch.concat(shallow_dino_features, dim=-1)
        shallow_dino_features = self.dino_projector(shallow_dino_features)
        shallow_dino_features = rearrange(shallow_dino_features, "b f h w c -> (b f) c h w")
        shallow_dino_features = F.interpolate(shallow_dino_features, (H, W), mode="bilinear", align_corners=True) 
        shallow_dino_features = rearrange(shallow_dino_features, "(b f) c h w -> b f h w c", b=b, f=f)            # (b, f, h, w, 1024*4)

        tokens_fused = []
        for i, token in enumerate(tokens):
            token = rearrange(token, "(b f) () d -> b f d", b=b, f=f)
            token_fused = self.token_projector_list[i](token)
            tokens_fused.append(rearrange(token_fused, "b f d -> (b f) () d"))

        global_tokens_fused = []
        for i, gtoken in enumerate(global_tokens):
            gtoken = rearrange(gtoken, "(b f) () d -> b f d", b=b, f=f)
            gtoken_fused = self.global_token_projector_list[i](gtoken)
            global_tokens_fused.append(rearrange(gtoken_fused, "b f d -> (b f) () d"))

        camera_tokens_fused = []
        for i, ctoken in enumerate(camera_tokens):
            ctoken = rearrange(ctoken, "(b f) () d -> b f d", b=b, f=f)
            ctoken_fused = self.token_projector_list[i](ctoken)
            camera_tokens_fused.append(rearrange(ctoken_fused, "b f d -> (b f) () d"))

        inputs = {}
        inputs["features"] = dino_features_fused         # [torch.Size([20, 30, 40, 1024])] * 4   # (420, 560)
        inputs["tokens"] = tokens_fused                  # [torch.Size([20, 1, 1024])] * 4
        inputs["global_tokens"] = global_tokens_fused    # [torch.Size([20, 1, 1024])] * 2
        inputs["camera_tokens"] = camera_tokens_fused    # [torch.Size([20, 1, 1024])] * 4
        inputs["image"] = rgbs                           # torch.Size([20, 3, 420, 560])

        outs = self.unidepth.pixel_decoder(inputs, {})
        if not ((h==H) and (w==W) and (ratio==1.0)):
            outs = _postprocess(outs, ratio, (H, W), mode=self.unidepth.interpolation_mode)

        # outs.keys() = dict_keys(['depth', 'confidence', 'depth_features', 'K'])
        # 'depth':            torch.Size([20, 1, 288, 384])
        # 'confidence':       torch.Size([20, 1, 288, 384])
        # 'depth_features':   torch.Size([20, 1200, 512])
        # 'K':                torch.Size([20, 3, 3])

        dpt_features = rearrange(outs['depth_features'], "b (h w) c -> b c h w", h=patch_h, w=patch_w)
        dpt_features = F.interpolate(dpt_features, size=(H, W), mode="bilinear", align_corners=True)
        dpt_features = rearrange(dpt_features, "(b f) c h w -> b f h w c", b=b, f=f)                    # torch.Size(b, f, 128, h, w)

        confidence_features = rearrange(outs['confidence'], "(b f) () h w -> b f h w ()", b=b, f=f)
        depths = rearrange(outs['depth'], "(b f) () h w -> b f h w", b=b, f=f)
        intrinsics = outs['K'].reshape(b, f, 3, 3).mean(axis=1)

        base_div = (H * W) ** 0.5
        focals = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1) / base_div
        principles = torch.stack([intrinsics[..., 0, 2] / W, intrinsics[..., 1, 2] / H], dim=-1)

        xy, _ = sample_image_grid((H, W), device)
        backward_weights = self.compute_correspondence_weights(
            self.grid_sample_features(earlier(dpt_features), xy + flows.backward),
            self.grid_sample_features(earlier(shallow_dino_features), xy + flows.backward),
            self.grid_sample_features(earlier(confidence_features), xy + flows.backward),
            later(dpt_features),
            later(shallow_dino_features),
            later(confidence_features)
        )

        return BackboneOutput(depths, backward_weights, (focals, principles))


    def compute_correspondence_weights(
        self,
        features_earlier: Float[Tensor, "batch frame height width channel"],
        dino_earlier: Float[Tensor, "batch frame height width channel"],
        conf_earlier: Float[Tensor, "batch frame height width channel"],
        features_later: Float[Tensor, "batch frame height width channel"],
        dino_later: Float[Tensor, "batch frame height width channel"],
        conf_later: Float[Tensor, "batch frame height width channel"]
    ) -> Float[Tensor, "batch frame height width"]:
        
        features = torch.cat((features_earlier, features_later, dino_earlier, dino_later, conf_earlier, conf_later), dim=-1)
        weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=0)
        return rearrange(weights, "b f h w () -> b f h w")


    def grid_sample_features(
        self,
        features: Float[Tensor, "batch frame height width channel"],
        grid: Float[Tensor, "batch frame height width xy=2"],
    ) -> Float[Tensor, "batch frame height width channel"]:
        b, f, _, _, _ = features.shape
        samples = F.grid_sample(
            rearrange(features, "b f h w c -> (b f) c h w"),
            rearrange(grid * 2 - 1, "b f h w xy -> (b f) h w xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rearrange(samples, "(b f) c h w -> b f h w c", b=b, f=f)
