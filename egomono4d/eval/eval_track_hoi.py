import torch
import numpy as np
import open3d as o3d
import pdb
import copy
import torch.nn.functional as F
from einops import einsum, rearrange
from ..model.procrustes import align_scaled_rigid

FLY_THRESHOLD = 0.05

def eval_track_hoi_conductor(pred_pcd, gt_pcd, gt_flys, hoi_masks, rgbs, tracker, commit=""):   # (b, f, h, w, 3) * 2, (b, f, h, w)

    b, f, h, w, _ = pred_pcd.shape

    # Align based on the first frame, the trajectory need to has the same scale with the whole scene.
    pred_pcd_align_base = pred_pcd[:, 0].reshape(b, h*w, 3)
    gt_pcd_align_base = gt_pcd[:, 0].reshape(b, h*w, 3)
    gt_flys_align_base = gt_flys[:, 0].reshape(b, h*w)

    delta_ext_scale, scale = align_scaled_rigid(pred_pcd_align_base, gt_pcd_align_base, gt_flys_align_base) 
    
    pred_pcd_align = pred_pcd.reshape(b, f*h*w, 3)
    gt_pcd_align = gt_pcd.reshape(b, f*h*w, 3)
    gt_flys_align = gt_flys.reshape(b, f*h*w)
    pred_pcd_align = torch.matmul(delta_ext_scale[:, :3,:3], pred_pcd_align.permute(0,2,1)).permute(0,2,1) + delta_ext_scale[:, :3, -1][:, None]

    ####################################### 3D Tracking ########################################

    gt_pcd_track = gt_pcd_align.reshape(b, f, h, w, 3)
    pred_pcd_track = pred_pcd_align.reshape(b, f, h, w, 3)
    if f < 5:
        rgbs = torch.concat([rgbs] + [rgbs[:, -1:]]*(5-f), dim=1)
    tracks, visibility = tracker(rgbs*255, segm_mask=hoi_masks, grid_size=35)

    rgbs = rgbs[:, :f]
    tracks_org = tracks[:, :f]
    visibility_org = visibility[:, :f]
    tracks_xy = tracks_org / torch.Tensor([w-1,h-1]).to(tracks.device)
    tracks_xy = rearrange(tracks_xy * 2 - 1, "b f p xy -> (b f) () p xy")

    mask_track = F.grid_sample(rearrange(gt_flys.float(), "b f h w -> (b f) () h w"),tracks_xy,mode="bilinear",padding_mode="border", align_corners=False,)
    mask_track = mask_track.reshape(b, f, -1)
    
    xyz_pred_track = F.grid_sample(rearrange(pred_pcd_track, "b f h w xyz -> (b f) xyz h w"),tracks_xy,mode="bilinear",padding_mode="border",align_corners=False,)
    xyz_pred_track = rearrange(xyz_pred_track, "(b f) xyz () p -> b f p xyz", b=b, f=f)
    xyz_gt_track = F.grid_sample(rearrange(gt_pcd_track, "b f h w xyz -> (b f) xyz h w"),tracks_xy,mode="bilinear",padding_mode="border",align_corners=False,)
    xyz_gt_track = rearrange(xyz_gt_track, "(b f) xyz () p -> b f p xyz", b=b, f=f)

    visibility = ((visibility_org * mask_track) == 1.0).float()
    visibility = (visibility.min(dim=-2, keepdim=True)[0] == 1.0).repeat(1,f,1).float()
    dis = torch.norm(xyz_pred_track - xyz_gt_track, dim=-1)          # (b, f, p)

    if (visibility[:, -1].sum() <= 0.0) or (visibility_org.shape[-1] == 0):
        track_fde = None
    else:
        track_fde = (dis[:, -1] * visibility[:, -1]).sum() / visibility[:, -1].sum()
        track_fde = 1000.0 * track_fde.item()

    if (visibility.sum() <= 0.0) or (visibility_org.shape[-1] == 0):
        track_ade = None
        apd_01, apd_025, apd_05 = None, None, None
        apd_1, apd_15, apd_2, apd_25 = None, None, None, None
        print("no valid points.")
    else:
        track_ade = (dis * visibility).sum() / visibility.sum()
        track_ade = 1000.0 * track_ade.item()
        apd_01 = 100.0 * (((dis < 0.01).float() * visibility).sum() / visibility.sum()).item()
        apd_025 = 100.0 * (((dis < 0.025).float() * visibility).sum() / visibility.sum()).item()
        apd_05 = 100.0 * (((dis < 0.05).float() * visibility).sum() / visibility.sum()).item()
        apd_1 = 100.0 * (((dis < 0.1).float() * visibility).sum() / visibility.sum()).item()
        apd_15 = 100.0 * (((dis < 0.15).float() * visibility).sum() / visibility.sum()).item()
        apd_2 = 100.0 * (((dis < 0.2).float() * visibility).sum() / visibility.sum()).item()
        apd_25 = 100.0 * (((dis < 0.25).float() * visibility).sum() / visibility.sum()).item()

    return {
        "Track3D_ADE(mm)": track_ade,
        "Track3D_FDE(mm)": track_fde,
        "APD_[.01](%)": apd_01,
        "APD_[.025](%)": apd_025,
        "APD_[.05](%)": apd_05,
        "APD_[.1](%)": apd_1,
        "APD_[.15](%)":  apd_15,
        "APD_[.2](%)": apd_2,
        "APD_[.25](%)": apd_25,
    }