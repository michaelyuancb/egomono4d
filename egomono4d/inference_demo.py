
import torch
import open3d as o3d
import numpy as np 
import pdb
import copy
import argparse
import time
import pickle
from tqdm import tqdm
from einops import einsum, rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

import os
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from cotracker.predictor import CoTrackerPredictor
from .config.common import get_typed_root_config
from .config.pretrain import PretrainCfg
from .model.model import Model
from .misc.data_util import compute_patch_cropped_shape
from .model.procrustes import align_scaled_rigid, align_rigid
from .misc.fly import detect_sequence_flying_pixels
from .model.model_wrapper_pretrain import ModelWrapperPretrain
from .model.projection import sample_image_grid, unproject, homogenize_points
from .dataset.dataset_custom import DatasetCustom

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLY_THRESHOLD = 0.03


def recover_pointclouds_sequence(depths, intrinsics, extrinsics, target_frame=0):
    f, h, w = depths.shape
    xy, _ = sample_image_grid((h, w), device=depths.device)
    gt_pcds_unp = unproject(xy, depths, rearrange(intrinsics, "f i j -> f () () i j"))

    extrinsics_source = rearrange(extrinsics, "fs i j -> fs () () i j")
    extrinsics_target = rearrange(extrinsics[target_frame:target_frame+1], "ft i j -> () ft () i j")
    relative_transformations = extrinsics_target.inverse() @ extrinsics_source

    pcds = einsum(
        relative_transformations,
        homogenize_points(gt_pcds_unp),
        "... i j, ... j -> ... i",
    )[..., :3]

    return pcds


def get_fly_from_noise_3d(xyzs, nb_neighbors=20, std_ratio=0.3):

    device = xyzs.device
    h, w, _ = xyzs.shape
    points = xyzs.reshape(-1, 3).cpu().numpy()
    kdtree = cKDTree(points)
    distances, _ = kdtree.query(points, k=nb_neighbors+1) 
    distances = distances[:, 1:] 
    mean_distances = np.mean(distances, axis=1)
    std_distances = np.std(mean_distances)
    mean_of_mean_distances = np.mean(mean_distances)
    threshold = mean_of_mean_distances + std_ratio * std_distances
    valid_indices = mean_distances > threshold

    return torch.Tensor(valid_indices.reshape(h, w)).to(device)


def get_final_point_clouds(ts_list, fly_masks, xyzs, rgbs, voxel_size=None):
    pcd_global = o3d.geometry.PointCloud()
    for i in ts_list:
        fly_msk = (1-fly_masks[i].reshape(-1)).astype(np.bool_)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs[i][fly_msk])
        pcd.colors = o3d.utility.Vector3dVector(rgbs[i][fly_msk])
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)
        pcd_global = pcd_global + pcd
    return pcd_global


def visualization(result, commit=""):
    f, h, w, _ = result['xyzs'].shape
    xyzs = np.array(result['xyzs'].reshape(f, -1, 3))
    rgbs = np.array(result['rgbs'].reshape(f, -1, 3))
    fly_masks = np.array(result['flys'].reshape(f, -1))

    # Save PointCloud Sequence
    ts_list = range(len(xyzs))
    if len(ts_list) > 25:
        step_size_10 = len(ts_list) // 25
        ts_list = range(0, len(xyzs), step_size_10)
    pcd = get_final_point_clouds(ts_list, fly_masks, xyzs, rgbs) 
    o3d.io.write_point_cloud(f"{commit}"+".ply", pcd)


@torch.no_grad()
def model_inference_demo(model_wrapper, dataloader, num_frames, filter_noise_3d=False):

    intrinsics_list, extrinsics_list, deps_list, rgbs_list = [], [], [], []
    weights_list, fly_masks_list = [], []
    device = 'cuda'

    ############################################ Prediction ##################################################
    for data in tqdm(dataloader):
        data['videos'] = data['videos'].to(device)
        model_output = model_wrapper.inference(data)

        rgbs_list.append(data['videos'].cpu())
        depth = model_output.depths.cpu()
        deps_list.append(depth)
        weights_list.append(model_output.backward_correspondence_weights.cpu())
        fly_masks = []
        for i in range(len(depth)):
            fl = detect_sequence_flying_pixels(np.array(depth[i]), threshold=FLY_THRESHOLD)
            fly_masks.append(fl)
        fly_masks_list.append(torch.Tensor(np.array(fly_masks)))
        intrinsics_list.append(model_output.intrinsics.cpu())
        extrinsics_list.append(model_output.extrinsics.cpu())
        
    ############################################ Alignment ##################################################
    intrinsics_list, extrinsics_list = torch.cat(intrinsics_list), torch.cat(extrinsics_list)
    weights_list, fly_masks_list = torch.cat(weights_list), torch.cat(fly_masks_list)
    deps_list, rgbs_list = torch.cat(deps_list), torch.cat(rgbs_list)
    xyzs, rgbs, deps, fly_masks, intrinsics = None, None, None, None, None

    seq_l_list = dataloader.dataset.seq_l_list
    print(f"seq_l: {seq_l_list}")
    last_seq_r = -1
    for i, seq_l in tqdm(enumerate(seq_l_list)):
        seq_r = seq_l + num_frames

        weight, videos, depths, fly_mask = weights_list[i], rgbs_list[i], deps_list[i], fly_masks_list[i]
        intrinsic, extrinsic = intrinsics_list[i], extrinsics_list[i]
        pcd_list = recover_pointclouds_sequence(depths, intrinsic, extrinsic)   # (f, n, 3)
        f, h, w = depths.shape
        rgb = videos.permute(0,2,3,1)

        if rgbs is not None:
            interval = last_seq_r - seq_l

            # align pointclouds & pcd_depths
            pcd_before, pcd_last = xyzs[-interval:].reshape(-1, 3), pcd_list[:interval].reshape(-1, 3)    
            weight_ = weight_msk[-interval:].reshape(-1)
            delta_ext_scale, scale = align_scaled_rigid(pcd_last, pcd_before, weights=weight_)   
            scale = scale.view(1, 1, 1)
            depths = depths * scale    

            # reproject pointclouds & flymasks
            intrinsic = intrinsic[interval:]
            pcd_list = torch.matmul(delta_ext_scale[:3,:3], pcd_list[interval:,:,:,:,None])[:,:,:,:,0] + delta_ext_scale[:3, 3][None,None,None,:]
            rgb = rgb[interval:]
            dep = depths[interval:]
            fly_masks[-interval:] = torch.logical_or(fly_masks[-interval:], fly_mask[:interval])
            fly_mask = fly_mask[interval:]

            # concat 
            xyzs, rgbs, deps = torch.concat([xyzs, pcd_list]), torch.concat([rgbs, rgb]), torch.concat([deps, dep])
            intrinsics = torch.concat([intrinsics, intrinsic])
            fly_masks, weight_msk = torch.concat([fly_masks, fly_mask]), torch.concat([weight_msk, weight[interval-1:]])

        else:
            xyzs, rgbs, intrinsics, deps = pcd_list, rgb, intrinsic, depths
            fly_masks, weight_msk = fly_mask, weight
        last_seq_r = seq_r

    if filter_noise_3d is True:
        for i in tqdm(range(f), desc="filter fly from 3d"):
            fly_3d = get_fly_from_noise_3d(xyzs[i])
            fly_masks[i] = fly_masks[i] * fly_3d

    pcds_org = recover_pointclouds_sequence(deps, intrinsics, torch.eye(4)[None])
    f = pcds_org.shape[0]
    extrinsics = align_rigid(pcds_org.reshape(f,-1,3), xyzs.reshape(f,-1,3), 1.0-fly_masks.reshape(f,-1))   # cam2world

    result = {
        "xyzs": xyzs,                            # (f, h, w, 3)
        "rgbs": rgbs,                            # (f, h, w, 3)
        "depths": deps,                          # (f, h, w),    depths used to recover point clouds. 
        "intrinsics": intrinsics,                # (f, 3, 3),    pixel2camera
        "extrinsics": extrinsics,                # (f, 4, 4),    camera2world
        "flys": fly_masks,                       # (f, h, w),    flys of depths
        "weights": weight_msk,                   # (f-1, h, w),   confident mask
    }
    return result


@torch.no_grad()
def inference_demo(args):
    
    ############################# Configuration #############################
    args.config = args.model_dir + "/.hydra/config.yaml"
    config = OmegaConf.load(args.config)
    cfg = get_typed_root_config(config, PretrainCfg)
    model_fp = args.model_dir + "/egomono4d"
    model_fp = model_fp + "/" + os.listdir(model_fp)[0] + "/checkpoints"
    model_list = os.listdir(model_fp)
    for i in range(len(model_list)):
        if model_list[i] not in ['last.ckpt']:
            model = model_list[i]
    model_fp = model_fp + "/" + model
    cfg.model.backbone.cache_dir = cfg.base_cache_dir
    cfg.flow.cache_dir = cfg.base_cache_dir
    print("Finish Prepare Configuration.")

    if args.windows_size is not None:
        cfg.preprocess.num_frames = args.windows_size

    ############################## DataLoader #############################
    dataset = DatasetCustom(cfg, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Finish Prepare DataLoader.")

    ############################## Model #############################
    print(f"Model File: {model_fp}.")
    patch_size = cfg.preprocess.patch_size
    patch_crop_shape = compute_patch_cropped_shape(cfg.preprocess.resize_shape, patch_size)
    num_frames = cfg.preprocess.num_frames
    model = Model(cfg.model, num_frames=num_frames, image_shape=patch_crop_shape,
                  patch_size=patch_size)
    model.to(DEVICE)
    model_wrapper = ModelWrapperPretrain.load_from_checkpoint(
        model_fp, cfg=cfg.model_wrapper, 
        cfg_flow=cfg.flow, model=model, device=DEVICE,
        cfg_track=None, losses=None, visualizers=None, enable_checkpoints_after=None
    )
    model_wrapper.eval()
    print("Finish Prepare Model.")

    ############################## Tracker & HOISeg ############################
    # checkpoint = "scaled_offline.pth"
    checkpoint = "cotracker2.pth"
    tracker = CoTrackerPredictor(checkpoint=cfg.base_cache_dir+"/cotracker_checkpoints/"+checkpoint)
    tracker = tracker.to(DEVICE)
    print("Finish Prepare Tracker.")
    os.makedirs(f"{cfg.base_cache_dir}/ego_hos_cache", exist_ok=True)

    commit = f"result_" + args.frames_dir.split('/')[-1]
    result = model_inference_demo(model_wrapper, dataloader, cfg.preprocess.num_frames, args.step_overlap)

    result_pkl = copy.deepcopy(result)
    for k in result_pkl.keys():
        result_pkl[k] = result_pkl[k].detach().cpu().numpy()
    pickle.dump(result_pkl, open(commit+".pkl", 'wb'))

    ############################## Visualization ############################

    visualization(result, commit=commit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference of 4D Dynamic Scene Reconstruction.")
    parser.add_argument("-s", "--step_overlap", type=int, default=1, help="how many step of interval for trajectory consistency calculation.")
    parser.add_argument("-w", "--windows_size", type=int, default=None, help="the windows size used for inference.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="the batch_size used in the inference stage.")
    parser.add_argument("-m", "--model_dir", type=str, default=None)
    parser.add_argument("-f", "--frames_dir", type=str, default=None)
    parser.add_argument("-c", "--cache_dir", type=str, default='./cache/data_custom', help="the cache dir to save the preprocess result of video frames.")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    inference_demo(args)

# INFER_MODE=True python -m egomono4d.inference_demo -m cache/models/ptr_all_350k -f examples/example_epic_kitchen