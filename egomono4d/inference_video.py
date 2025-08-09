
import torch
from .dataset import get_dataset
import open3d as o3d
import numpy as np 
import pdb
import cv2
import hydra
import copy
from PIL import Image
import argparse
import time
from tqdm import tqdm
from einops import einsum, rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from .model.procrustes import align_scaled_rigid
from torch.utils.tensorboard import SummaryWriter


import os
from einops import einsum, rearrange
from omegaconf import OmegaConf
from hydra.experimental import initialize_config_dir, compose
import cotracker
from cotracker.predictor import CoTrackerPredictor

from .utils import save_pickle, batch_recover_pointclouds_sequence
from .loss import get_losses
from .misc.data_util import compute_patch_cropped_shape
from .visualization import get_visualizers, VisualizerCoTracker
from .config.common import get_typed_root_config
from .config.pretrain import PretrainCfg
from .model.model import Model
from .misc.fly import detect_sequence_flying_pixels
from .model.model_wrapper_pretrain import ModelWrapperPretrain
from .model.procrustes import align_rigid_unweighted


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLY_THRESHOLD = 0.03


def get_final_point_clouds(st, ed, fly_masks, xyzs, rgbs):
    pcd_xyz, pcd_rgb = [], []
    # pdb.set_trace()
    for i in range(st, ed):
        fly_msk = (1-fly_masks[i].reshape(-1)).astype(np.bool_)
        pcd_xyz.append(xyzs[i][fly_msk])
        pcd_rgb.append(rgbs[i][fly_msk])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcd_xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(pcd_rgb, axis=0))
    return pcd


def flow_visualization(rgb, xyz, video, fly_mask, tracker, save_fp="flow_vis.pkl", grid_size=50, device='cpu'):

    # rgb:       f, n, 3
    # xyz:       f, n, 3
    # video:     f, h, w, 3
    # fly_mask:  f, h, w

    f, h, w, _ = video.shape
    b = 1
    if f < 5:
        # satisfy the co-tracker demands. 
        video = np.concatenate([video] + [video[-1:]]*(5-f), axis=0)

    video_tensor = torch.tensor(video).permute(0, 3, 1, 2)[None].float().to(device)  # (1,f,c,h,w)
    pred_tracks, pred_visibility = tracker(video_tensor*255, grid_size=grid_size)
    
    video = video[:f]
    pred_tracks = pred_tracks[:, :f]
    pred_visibility = pred_visibility[:, :f]
    pred_tracks_xy = pred_tracks / torch.Tensor([w-1,h-1]).to(device)
    vis = VisualizerCoTracker(pad_value=120, linewidth=1, save_dir='.')
    vis.visualize(video_tensor*255, pred_tracks_xy, pred_visibility, filename='.'.join(save_fp.split('.')[:-1]))
   
    pred_visibility = pred_visibility.min(dim=1,keepdim=True)[0]                   # b,1,n
    pred_tracks = pred_tracks[pred_visibility.repeat(1,f,1)].reshape(b,f,-1,2)     # b, f, n
    xyz = xyz.reshape(f, h, w, 3)                                  # f, h, w, 3
    xyz_tensor = torch.Tensor(xyz)[None].to(device)

    fly_mask_tensor = torch.tensor(fly_mask[None]).to(device)   # (b, f, h, w)
    xyz_tensor[fly_mask_tensor == 1] = float('nan')

    flow = F.grid_sample(
        rearrange(xyz_tensor, "b f h w xyz -> (b f) xyz h w"),
        rearrange(pred_tracks_xy * 2 - 1, "b f p xyz -> (b f) () p xyz"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    # pdb.set_trace()
    flow = rearrange(flow, "(b f) xyz () p -> b f p xyz", b=b, f=f)    # (b, f, n, 3)
    valid_mask = torch.isnan(flow).sum(axis=[1,3]) == 0
    valid_flow = flow[valid_mask[:,None].repeat(1,f,1)]
    valid_flow = valid_flow.cpu().numpy().reshape(f,-1,3)

    flow_vis_result = {
        "video_rgb": rgb.reshape(f, -1, 3),
        "video_xyz": xyz.reshape(f, -1, 3),
        "video_fly_mask": fly_mask.reshape(f, -1),
        "flow": valid_flow
    }
    save_pickle(save_fp, flow_vis_result)


def generate_video(xyzs, rgbs, intrinsics, data_all, shape, camera_base=np.array([0.0,0.0,0.0])):
    video_frames = []
    h, w = shape
    print("generate projection 2d videos.")
    for i in tqdm(range(len(xyzs))):
        xyz = np.array(xyzs[i]) + camera_base
        rgb = np.array(rgbs[i])
        image = np.zeros((h, w, 3), dtype=np.uint8)
        projected_points = (intrinsics[0] @ xyz.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        for (x, y), color in zip(projected_points, rgb):
            x, y = int(x * (w-1)), int(y * (h-1))
            if 0 <= x < w and 0 <= y < h:
                image[y, x] = (color * 255).astype(np.uint8)
        video_frames.append(image)

    # pdb.set_trace()
    out = cv2.VideoWriter('origin.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
    for color in data_all['videos'][0]:
        out.write(cv2.cvtColor(np.array(color.cpu() * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    out.release()

    out = cv2.VideoWriter('output_f0.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    create_merge_video('origin.mp4', 'output_f0.mp4', "origin", "frame0_view", "visualization_frame0_view.mp4")


def create_merge_video(v1_fp, v2_fp, v1_title, v2_title, output_fp):
    cap1 = cv2.VideoCapture(v1_fp)
    cap2 = cv2.VideoCapture(v2_fp)

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert height1 == height2, "The heights of the videos must be the same"

    blank_width = 50
    output_width = width1 + width2 + 3 * blank_width
    output_height = height1 + 100  # 增加高度用于标题
    out = cv2.VideoWriter(output_fp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        frame1_bgr = frame1
        frame2_bgr = frame2
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined_frame[100:100+height1, blank_width:blank_width+width1] = frame1_bgr
        combined_frame[100:100+height1, 2*blank_width+width1:2*blank_width+width1+width2] = frame2_bgr
        cv2.putText(combined_frame, v1_title, (blank_width + width1 // 2 - 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, v2_title, (2*blank_width + width1 + width2 // 2 - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()


@torch.no_grad()
def model_inference_conductor(data, model_wrapper, num_frames, step_overlap, tracker=None, vis=False):
    n_images = data['videos'].shape[1]
    seq_l_last = (n_images-num_frames) // (num_frames-step_overlap)
    seq_l_list = [i*(num_frames-step_overlap) for i in range(seq_l_last+1)]
    if seq_l_list[-1] + num_frames < n_images:
        seq_l_list.append(n_images - num_frames)
    data_clip = copy.deepcopy(data)

    intrinsics_list, extrinsics_list = [], []
    deps_list, rgbs_list = [], []
    weights_list, fly_masks_list, vis_list = [], [], []

    seq_iter = tqdm(enumerate(seq_l_list)) if vis is True else enumerate(seq_l_list)
    for i, seq_l in seq_iter:
        seq_r = seq_l + num_frames
        for k, v in data.items():
            if type(data[k]) is list: 
                continue
            data_clip[k] = data[k][:, seq_l:seq_r]

        model_output = model_wrapper.inference_step(data_clip, vis=False)

        rgbs_list.append(data_clip['videos'].cpu())
        depth = model_output.depths.cpu()
        deps_list.append(depth)
        weights_list.append(model_output.backward_correspondence_weights.cpu())
        fly_masks = []
        for i in range(len(depth)):
            fl = detect_sequence_flying_pixels(np.array(depth[i]), threshold=FLY_THRESHOLD)
            fly_masks.append(fl)
        fly_masks_list.append(torch.Tensor(fly_masks))
        intrinsics_list.append(model_output.intrinsics.cpu())
        extrinsics_list.append(model_output.extrinsics.cpu())

    ######################################## Sequence Alignment ######################################

    intrinsics = None
    xyzs, rgbs, deps, fly_masks = None, None, None, None

    seq_iter = tqdm(enumerate(seq_l_list)) if vis is True else enumerate(seq_l_list)
    last_seq_r = -1
    for i, seq_l in seq_iter:
        seq_r = seq_l + num_frames

        weight = weights_list[i]
        videos = rgbs_list[i]
        depths = deps_list[i]
        fly_mask = fly_masks_list[i]
        intrinsic = intrinsics_list[i]
        extrinsic = extrinsics_list[i]
        pcd_list = batch_recover_pointclouds_sequence(depths, intrinsic, extrinsic)   # (b, f, n, 3)
        b, f, h, w = depths.shape
        rgb = videos.permute(0,1,3,4,2)

        if rgbs is not None:
            interval = last_seq_r - seq_l

            # align pointclouds
            pcd_before = xyzs[:, -interval:].reshape(b, -1, 3)   
            pcd_last = pcd_list[:, :interval].reshape(b, -1, 3)       
            weight_ = weight_msk[:, -interval:].reshape(b, -1)
            delta_ext_scale, scale = align_scaled_rigid(pcd_last, pcd_before, weights=weight_)             # (4, 4)
            scale = scale.view(-1, 1, 1, 1)
            depth = depth * scale
           
            # reproject pointclouds & flymasks
            intrinsic = intrinsic[:, interval:]
            pcd_list = torch.matmul(delta_ext_scale[:, None, None, None, :3,:3], pcd_list[:,interval:,:,:,:,None])[:,:,:,:,:,0] + delta_ext_scale[:, :3, 3][:,None,None,None,:]
            rgb = rgb[:, interval:]
            fly_masks[:, -interval:] = torch.logical_or(fly_masks[:, -interval:], fly_mask[:, :interval])
            fly_mask = fly_mask[:, interval:]

            # concat 
            xyzs = torch.concat([xyzs, pcd_list], dim=1)
            rgbs = torch.concat([rgbs, rgb], dim=1)
            deps = torch.concat([deps, depths[:, interval:]], dim=1)
            intrinsics = torch.concat([intrinsics, intrinsic], dim=1)
            fly_masks = torch.concat([fly_masks, fly_mask], dim=1)
            weight_msk = torch.concat([weight_msk, weight[:, interval-1:]], dim=1)

        else:
            xyzs, rgbs, deps, intrinsics = pcd_list, rgb, depths, intrinsic
            fly_masks, weight_msk = fly_mask, weight

        last_seq_r = seq_r
    
    pcds_org = batch_recover_pointclouds_sequence(deps, intrinsics, torch.eye(4)[None, None])
    (b, f, h, w, _) = pcds_org.shape
    extrinsics = align_rigid_unweighted(pcds_org.reshape(b,f,-1,3), xyzs.reshape(b,f,-1,3))   # cam2world

    result = {
        "xyzs": xyzs,                  # (b, f, h, w, 3)
        "rgbs": rgbs,                  # (b, f, 3, h, w)
        "depths": deps,                # (b, f, h, w)
        "intrinsics": intrinsics,      # (b, f, 3, 3)
        "extrinsics": extrinsics,      # (b, f, 4, 4)
        "flys": fly_masks,             # (b, f, h, w)
        "weights": weight_msk,         # (b, f-1, h, w)
        "seq_base": seq_l_list
    }

    return result


def put_data_batch(data):
    for k, v in data.items():
        try:
            data[k] = v[None]
        except:
            data[k] = [v]


def put_data_device(data, device):
    for k, v in data.items():
        try:
            data[k] = v.to(device)
        except:
            pass



@torch.no_grad()
def model_result_inference(model_wrapper, dataloader, tracker, num_frames, step_overlap, commit=""):

    n_vis_list = range(0, 1)
    for vis_i in n_vis_list: 

        data_all = dataloader.dataset[vis_i]
        put_data_batch(data_all)
        
        scene = data_all['datasets'][0]
        print(f"visualize Data {vis_i}, idx={data_all['scenes'][0]}")
        put_data_device(data_all, DEVICE)
        assert step_overlap <= num_frames

        result = model_inference_conductor(data_all, model_wrapper, num_frames, step_overlap, tracker=tracker, vis=True)
        batch_idx = 0

        b, f, h, w, _ = result['xyzs'].shape
        xyzs = np.array(result['xyzs'][batch_idx].reshape(f, -1, 3).cpu())
        rgbs = np.array(result['rgbs'][batch_idx].reshape(f, -1, 3).cpu())
        fly_masks = np.array(result['flys'][batch_idx].reshape(f, -1).cpu())
        intrinsics = np.array(result['intrinsics'][batch_idx].reshape(f, 3, 3).cpu())

        # Save PointCloud Sequence
        pdb.set_trace()
        st,ed = 0,len(xyzs)
        pcd = get_final_point_clouds(st, ed, fly_masks, xyzs, rgbs) 
        o3d.io.write_point_cloud(f"tmp_pcd_{commit}_vs{vis_i}_{scene}_{st}_{ed}"+".ply", pcd)

        xyzs_unproj = batch_recover_pointclouds_sequence(result['deps'], result['intrinsics'], result['extrinsics'])
        pcds_unproj = get_final_point_clouds(st, ed, fly_masks, xyzs_unproj.reshape(len(xyzs), -1, 3), rgbs) 
        o3d.io.write_point_cloud(f"tmp_pcd_unproj_{commit}_vs{vis_i}_{scene}_{st}_{ed}"+".ply", pcds_unproj)

        st,ed = 0, len(xyzs)
        n_pt = len(xyzs[0])
        video_c = np.array(data_all['videos'][batch_idx].cpu()).transpose(0, 2, 3, 1)[st:ed]   # (f,h,w,c)
        xyz_c = np.concatenate(xyzs, axis=0)[st*n_pt:ed*n_pt].reshape(-1, n_pt, 3)              # (f,n,3)
        rgb_c = np.concatenate(rgbs, axis=0)[st*n_pt:ed*n_pt].reshape(-1, n_pt, 3)              # (f,n,3)
        fly_mask_c = np.stack(fly_masks[st:ed] , axis=0)                                  # (f,h,w)
        flow_visualization(rgb_c, xyz_c, video_c, fly_mask_c.reshape(f,h,w), tracker, grid_size=35, save_fp=f"tmp_flow_{commit}_vs{vis_i}_{scene}_{st}_{ed}.pkl", device=DEVICE)
        
    pdb.set_trace()


@torch.no_grad()
def inference(cfg, fp=""):
    
    cfg.model.backbone.cache_dir = cfg.base_cache_dir
    cfg.flow.cache_dir = cfg.base_cache_dir
    cfg.tracking.cache_dir = cfg.base_cache_dir
    for dataset_cfg in cfg.dataset:
        dataset_cfg.resize_shape = cfg.preprocess.resize_shape
        dataset_cfg.patch_size = cfg.preprocess.patch_size
        dataset_cfg.num_frames = cfg.preprocess.num_frames
        dataset_cfg.cache_dir = cfg.base_cache_dir
        if hasattr(dataset_cfg, "mask_flow_model"):
            dataset_cfg.mask_flow_model = cfg.flow 
        dataset_cfg.all_frames = True

    patch_size = cfg.preprocess.patch_size
    patch_crop_shape = compute_patch_cropped_shape(cfg.preprocess.resize_shape, patch_size)
    num_frames = cfg.preprocess.num_frames

    model = Model(cfg.model, num_frames=num_frames, image_shape=patch_crop_shape,
                  patch_size=patch_size)
    model.to(DEVICE)
    losses = get_losses(cfg.loss)
    visualizers = get_visualizers(cfg.visualizer)
    checkpoint = "cotracker2.pth"
    tracker = CoTrackerPredictor(checkpoint=cfg.base_cache_dir+"/cotracker_checkpoints/"+checkpoint)
    tracker = tracker.to(DEVICE)

    model_wrapper = ModelWrapperPretrain.load_from_checkpoint(
        args.model,
        cfg=cfg.model_wrapper,
        cfg_flow=cfg.flow,
        cfg_track=cfg.tracking,
        model=model,
        losses=losses,
        visualizers=visualizers,
        enable_checkpoints_after=None,
        device=DEVICE
    )
    model_wrapper.eval()

    dataloader = DataLoader(get_dataset(cfg.dataset, "test", global_rank=0, world_size=1), batch_size=1)
    model_result_inference(model_wrapper, dataloader, tracker, cfg.preprocess.num_frames, args.step_overlap, commit=fp)


def get_cfg(args):
    if args.config is None:
        args.config = args.folder_path + "/.hydra/config.yaml"
        config = OmegaConf.load(args.config)
        cfg = get_typed_root_config(config, PretrainCfg)    
    else:
        sp_file = args.config.split('/')
        initialize_config_dir(config_dir='/'.join(sp_file[:-1]), job_name="dfm_eval")
        cfg = compose(config_name='.'.join(sp_file[-1].split('.')[:-1]))
        cfg = get_typed_root_config(cfg, PretrainCfg)    

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference of 4D Dynamic Scene Reconstruction.")
    parser.add_argument("-s", "--step_overlap", type=int, default=1, help="how many step of interval for trajectory consistency calculation.")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-f", "--folder_path", type=str, default=None)
    args = parser.parse_args()
    if args.folder_path is not None:
        cfg = get_cfg(args)

        args.model = args.folder_path 
        flist = os.listdir(args.model)
        for fp in flist:
            if fp.startswith("train_ddp"):
                continue
            if fp == ".hydra" or fp == "pretrain.log": 
                continue
            args.model = args.model + "/" + fp + "/dfm"
            args.model = args.model + "/" + os.listdir(args.model)[0] + "/checkpoints"
            model_list = os.listdir(args.model)
            model_select = 'last.ckpt'
            for i in range(len(model_list)):
                if model_list[i] not in ['last.ckpt']:
                    model = model_list[i]
                    print(f"select model {model_list[i]}")
                    break
            args.model = args.model + "/" + model
            break
        # pdb.set_trace()
    inference(cfg, fp)