import torch
import cv2
import pdb
import os
import json
import numpy as np 
from PIL import Image
from unidepth.models import UniDepthV2

def get_depth_estimator(estimator_name="unidepth_v2_large", cache_dir='.cache', device='cuda'):
    # pdb.set_trace()
    if estimator_name in ["unidepth_v2_large", "unidepth_v2_small"]:
        version = 'v2'
        if estimator_name.endswith('large'):
            backbone = 'vitl'
        elif estimator_name.endswith('small'):
            backbone = 'vits'
        with open(os.path.join(cache_dir, "unidepth_v2_checkpoints", f"unidepth-{version}-{backbone}14.json")) as f:
            config = json.load(f)
        model = UniDepthV2(config)
        model_dir = os.path.join(cache_dir, "unidepth_v2_checkpoints", f"unidepth-{version}-{backbone}14.bin")
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
        model = model.to(device).eval()
        return model
    else:
        raise ValueError(f"Unsupport Depth Estimator: {estimator_name}. Supportion: [depth_anything_v2_large].")



def estimate_relative_depth(pil_image: Image.Image, 
                            model,
                            estimator_name="unidepth_v2_large"):
    if estimator_name in ['unidepth_v2_large', 'unidepth_v2_small']:
        rgb = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1) # C, H, W
        predictions = model.infer(rgb)
        predictions['depth'] = predictions['depth'].cpu().detach().numpy()[0,0]
        predictions['intrinsics'] = predictions['intrinsics'].cpu().detach().numpy()[0]
        predictions['points'] = predictions['points'].cpu().detach().numpy()[0].transpose(1,2,0)
        return predictions
    else:
        raise ValueError("Unsupport Disparity-Depth Estimator: {estimator_name}. Supportion: [depth_anything_v2_large, depth_anything_v2_large_indoor].")


def save_estimate_disparity_png(e_dep, e_dep_fp_img):
    # black: 0    <---->    white: 1
    # we follow that more closer to camera, more closer to white color. 
    e_dep = (255 * (e_dep - e_dep.min()) / (e_dep.max() - e_dep.min())).astype(np.uint8)
    e_dep_img = Image.fromarray(e_dep, mode='L')
    e_dep_img.save(e_dep_fp_img)


def save_estimate_depth_png(e_dep, e_dep_fp_img):
    # black: 1 (deeper)   <---->    white: 0 (closer)
    e_dep = np.log(e_dep)           # for depth we first conduct log for it.
    dp_norm = (e_dep - e_dep.min()) / (e_dep.max() - e_dep.min())
    e_dep = 255 * (1.0 - dp_norm)
    if e_dep.dtype != np.uint8:
        e_dep = e_dep.astype(np.uint8)
    e_dep_img = Image.fromarray(e_dep, mode='L')
    e_dep_img.save(e_dep_fp_img)
