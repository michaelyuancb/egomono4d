import torch
import pdb
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.core.transformations import quaternion_from_matrix
from evo.core import metrics
from evo.tools import file_interface
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe


def tensor_to_trajectory(tensor):
    # Tensor(seq_length, 4, 4) --> PoseTrajectory3D
    seq_length = len(tensor)
    timestamps = np.arange(seq_length) 
    return PoseTrajectory3D(poses_se3=list(tensor.cpu().numpy()), timestamps=timestamps)
    

def eval_extrinsic_conductor(pred_extrinsic, gt_extrinsic, correct_scale=True):  # for mono-slam, correct_scale=True

    ate_list, rpe_trans_list, rpe_rot_list = [], [], []
    for i in range(len(pred_extrinsic)):

        pred = pred_extrinsic[i]
        gt = gt_extrinsic[i]
        traj_est = tensor_to_trajectory(pred)
        traj_ref = tensor_to_trajectory(gt)

        ate_result = main_ape.ape(traj_ref, traj_est, est_name='ate',
                                  pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=correct_scale)

        rpe_trans_result = main_rpe.rpe(traj_ref, traj_est, est_name='rpe_t', delta=1.0, delta_unit=metrics.Unit.frames,
                                        pose_relation=metrics.PoseRelation.translation_part, align=True, rel_delta_tol=0.1, correct_scale=correct_scale)

        rpe_rot_result = main_rpe.rpe(traj_ref, traj_est, est_name='rpe_r', delta=1.0, delta_unit=metrics.Unit.frames, 
                                      pose_relation=metrics.PoseRelation.rotation_angle_deg, align=True, rel_delta_tol=0.1, correct_scale=correct_scale)
        ate_list.append(ate_result.stats["mean"])
        rpe_trans_list.append(rpe_trans_result.stats["mean"])
        rpe_rot_list.append(rpe_rot_result.stats["mean"])

    return {
        'CAM_ATE(mm)': 1000.0 * sum(ate_list) / len(ate_list),
        'CAM_RPE_Trans(mm)': 1000.0 * sum(rpe_trans_list) / len(rpe_trans_list),
        'CAM_RPE_Rot(deg)': sum(rpe_rot_list) / len(rpe_rot_list)
    }

