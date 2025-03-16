import torch
import pdb
from ..loss.loss_midas import compute_scale_and_shift

EPS = 1e-6

def eval_depth_conductor(pred_depth, gt_depth, gt_flys):   # (b, f, h, w) * 3

    b, f, h, w = pred_depth.shape

    pred_depth_align = pred_depth.reshape(b,f*h,w)
    gt_depth_align = gt_depth.reshape(b,f*h,w)
    gt_flys_align = gt_flys.reshape(b,f*h,w)
    scale_video, shift_video = compute_scale_and_shift(pred_depth_align, gt_depth_align, gt_flys_align)

    pred_depth_align = scale_video.view(-1, 1, 1) * pred_depth_align + shift_video.view(-1, 1, 1)
    err = torch.abs(pred_depth_align - gt_depth_align)
    err_rel = err / (gt_depth_align + EPS)

    err_sq = err ** 2
    thresh = torch.maximum((gt_depth_align / (pred_depth_align + EPS)), (pred_depth_align / (gt_depth_align + EPS)))
    gt_flys = gt_flys.reshape(b,f*h,w)

    return {
        'DEPTH_AbsRel(%)': 100 * ((err_rel*gt_flys).sum()/(gt_flys.sum())).item(),
        'DEPTH_RMSE(mm)': 1000 * torch.sqrt((err_sq*gt_flys).sum()/(gt_flys.sum())).item(),
        'DEPTH_Delta_[.025](%)': (100*(((thresh < 1.025).float()*gt_flys).sum())/(gt_flys.sum())).item(),
        'DEPTH_Delta_[.05](%)': (100*(((thresh < 1.05).float()*gt_flys).sum())/(gt_flys.sum())).item(),
        'DEPTH_Delta_[.1](%)': (100*(((thresh < 1.1).float()*gt_flys).sum())/(gt_flys.sum())).item(),
        'DEPTH_Delta_[.25](%)': (100*(((thresh < 1.25).float()*gt_flys).sum())/(gt_flys.sum())).item(),
        'DEPTH_Delta_[.25]^2(%)': (100*(((thresh < 1.25**2).float()*gt_flys).sum())/(gt_flys.sum())).item(),
    }