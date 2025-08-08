import torch
import numpy as np
import open3d as o3d
import pdb
import copy
import torch.nn.functional as F
from einops import einsum, rearrange
from ..model.procrustes import align_scaled_rigid
import kaolin as kal

FLY_THRESHOLD = 0.05

def eval_pointcloud_conductor(pred_pcd, gt_pcd, gt_flys, rgbs, commit=""):   # (b, f, h, w, 3) * 2, (b, f, h, w)
    """ An implementation based on Kaolin library. """
    
    b, f, h, w, _ = pred_pcd.shape

    pred_pcd_align = pred_pcd.reshape(b, f*h*w, 3)
    gt_pcd_align = gt_pcd.reshape(b, f*h*w, 3)
    gt_flys_align = gt_flys.reshape(b, f*h*w)

    delta_ext_scale, scale = align_scaled_rigid(pred_pcd_align, gt_pcd_align, gt_flys_align) 
    pred_pcd_align = torch.matmul(delta_ext_scale[:, :3,:3], pred_pcd_align.permute(0,2,1)).permute(0,2,1) + delta_ext_scale[:, :3, -1][:, None]

    pred_pcd_align = pred_pcd_align.reshape(b, f, h*w, 3)
    gt_pcd_align = gt_pcd_align.reshape(b, f, h*w, 3)
    gt_flys_align = gt_flys_align.reshape(b, f, h*w)

    ######################################## VIS ###############################################
    # vis_flys = gt_flys_align.reshape(b, -1)
    # rgbss = rgbs.permute(0,1,3,4,2)
    # rgbss = rgbss.reshape(b, -1, 3)[vis_flys == 1]
    # ppa = pred_pcd_align.reshape(b, -1, 3)[vis_flys == 1].reshape(-1, 3)
    # pcd = o3d.geometry.PointCloud() 
    # pcd.points = o3d.utility.Vector3dVector(np.array(ppa.cpu().detach().reshape(-1, 3)))
    # pcd.colors = o3d.utility.Vector3dVector(np.array(rgbss.cpu().detach().reshape(-1, 3)))
    # o3d.io.write_point_cloud(f"pcd_pred_{commit}"+".ply", pcd)
    
    # pcd = o3d.geometry.PointCloud()
    # gpa = gt_pcd_align.reshape(b, -1, 3)[vis_flys == 1].reshape(-1, 3)
    # pcd.points = o3d.utility.Vector3dVector(np.array(gpa.cpu().detach().reshape(-1, 3)))
    # pcd.colors = o3d.utility.Vector3dVector(np.array(rgbss.cpu().detach().reshape(-1, 3)))
    # o3d.io.write_point_cloud(f"pcd_gt_{commit}"+".ply", pcd)
    ######################################## VIS ###############################################
    
    cds, f001, f0025, f005, f01 = [], [], [], [], []
    n_f = pred_pcd_align.shape[0]
    for i in range(n_f):
        gt_fly_f = gt_flys_align[:, i]
        pred_pcd_f = pred_pcd_align[:, i][gt_fly_f == 1][None]
        gt_pcd_f = gt_pcd_align[:, i][gt_fly_f == 1][None]
        cd = kal.metrics.pointcloud.chamfer_distance(pred_pcd_f, gt_pcd_f)
        cds.append(cd)
        f001.append(kal.metrics.pointcloud.f_score(pred_pcd_f, gt_pcd_f, radius=0.01))
        f0025.append(kal.metrics.pointcloud.f_score(pred_pcd_f, gt_pcd_f, radius=0.025)) 
        f005.append(kal.metrics.pointcloud.f_score(pred_pcd_f, gt_pcd_f, radius=0.05))
        f01.append(kal.metrics.pointcloud.f_score(pred_pcd_f, gt_pcd_f, radius=0.1))

    cd = sum(cds) / (b*n_f)
    f_score_001 = sum(f001) / (b*n_f)
    f_score_0025 = sum(f0025) / (b*n_f)
    f_score_005 = sum(f005) / (b*n_f)
    f_score_01 = sum(f01) / (b*n_f)

    return {
        "PCD_ChamferDistance(mm)": 1000.0 * cd.item(),
        "PCD_FScore_[.01]": 100.0*f_score_001.item(),
        "PCD_FScore_[.025]": 100.0*f_score_0025.item(),
        "PCD_FScore_[.05]": 100.0*f_score_005.item(),
        "PCD_FScore_[.1]": 100.0*f_score_01.item(),
    }
