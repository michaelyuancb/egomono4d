from dataclasses import dataclass, replace

import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from PIL import Image
import numpy as np
import pdb
import os
import torch
from torch import Tensor
from typing import Union

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    import open3d as o3d


@dataclass
class PreProcessingCfg:
    resize_shape: Union[tuple, int] = (300, 400)
    patch_size: int = 14
    num_frames: int = 4


def compute_patch_cropped_shape(
    shape: tuple,
    patch_size: int,
) -> tuple:
    h, w = shape

    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size
    return h_new, w_new


def pil_resize_to_center_crop(
    image: Image.Image,
    resize_shape: tuple,
    cropped_shape: tuple,
    depth_process=False
):  # -> tuple[
    #    Image.Image,  # the image itself
    #    tuple[int, int],  # image shape after scaling, before cropping
    
    w_old, h_old = image.size
    h_new, w_new = resize_shape
    h_crp, w_crp = cropped_shape

    # Figure out the scale factor needed to cover the desired shape with a uniformly
    # scaled version of the input image. Then, resize the input image.
    scale_factor = max(h_new / h_old, w_new / w_old)
    h_scaled = round(h_old * scale_factor)
    w_scaled = round(w_old * scale_factor)
    if depth_process is True:
        image_scaled = image.resize((w_scaled, h_scaled), Image.NEAREST)
    else:
        image_scaled = image.resize((w_scaled, h_scaled), Image.LANCZOS)

    # Center-crop the image.
    x = (w_scaled - w_crp) // 2
    y = (h_scaled - h_crp) // 2
    image_cropped = image_scaled.crop((x, y, x + w_crp, y + h_crp))
    return image_cropped, (h_scaled, w_scaled)


def resize_crop_intrinisic(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origin_shape: tuple,
    scaled_shape: tuple,
    croped_shape: tuple
):
    h_old, w_old = origin_shape
    h_scl, w_scl = scaled_shape
    h_new, w_new = croped_shape

    # reshape updatation
    sx = w_scl / w_old
    sy = h_scl / h_old
    new_intrinsics = intrinsics.clone()
    new_intrinsics[..., 0, 0] *= sx
    new_intrinsics[..., 0, 2] *= sx
    new_intrinsics[..., 1, 1] *= sy
    new_intrinsics[..., 1, 2] *= sy

    # center_crop updataion
    offset_x = (w_scl - w_new) / 2
    offset_y = (h_scl - h_new) / 2
    new_intrinsics[0, 2] -= offset_x  
    new_intrinsics[1, 2] -= offset_y

    return new_intrinsics


def canonicalize_intrinisic(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple
):
    # NOTE: (michael) Intrinsic Canonicalization to (1,1) size space for mixture dataset training. 
    h, w = shape
    new_intrinsics = intrinsics.clone()
    new_intrinsics[..., 0, 0] = new_intrinsics[..., 0, 0] / w
    new_intrinsics[..., 0, 2] = new_intrinsics[..., 0, 2] / w 
    new_intrinsics[..., 1, 1] = new_intrinsics[..., 1, 1] / h
    new_intrinsics[..., 1, 2] = new_intrinsics[..., 1, 2] / h 
    return new_intrinsics


def visualize_pcd_from_rgbd_fp(rgb_fp, depth_fp, intrinsic):
    color = o3d.io.read_image(rgb_fp)
    depth = o3d.io.read_image(depth_fp)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)
    camera = o3d.camera.PinholeCameraIntrinsic()
    H, W, _ = np.asarray(color).shape
    camera.set_intrinsics(W, H, intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.025)
    return voxel_down_pcd