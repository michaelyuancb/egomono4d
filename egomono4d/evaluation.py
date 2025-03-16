
import torch
from .dataset import get_dataset
import numpy as np 
import pdb
import time
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from .misc.fly import detect_sequence_flying_pixels

import open3d as o3d
import os
from cotracker.predictor import CoTrackerPredictor
from .loss import get_losses
from .visualization import get_visualizers
from .model.model import Model

from .model.model_wrapper_pretrain import ModelWrapperPretrain

from .eval import eval_depth_conductor, eval_pointcloud_conductor, eval_extrinsic_conductor, eval_track_conductor, eval_track_hoi_conductor
from .inference_video import put_data_device, model_inference_conductor, batch_recover_pointclouds_sequence, get_cfg


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE={DEVICE}")
FLY_THRESHOLD = 0.05
EPS = 1e-6
earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa
SAVE_GT = False


def model_result_evaluate(base_save_dir, dataset_name, model_name, args, cfg, model_wrapper, dataloader, tracker):
    

    model_name = f"{model_name}_s{args.step_overlap}_t{args.inference_trunc}"  # _iv12
    if args.windows_size is not None:
        model_name = model_name + f"_win{args.windows_size}"
    egomono4d_save_dir = os.path.join(base_save_dir, model_name)
    os.makedirs(egomono4d_save_dir, exist_ok=True)
    egomono4d_save_dir = os.path.join(egomono4d_save_dir, dataset_name)
    os.makedirs(egomono4d_save_dir, exist_ok=True)

    gt_save_dir = os.path.join(base_save_dir, "gt")
    os.makedirs(gt_save_dir, exist_ok=True)
    gt_save_dir = os.path.join(gt_save_dir, dataset_name)
    os.makedirs(gt_save_dir, exist_ok=True)
    
    step_overlap = args.step_overlap
    num_frames = cfg.preprocess.num_frames

    ##################################### Inference Result #######################################
    data_id = 0
    gt_data_id = 0 
    n_batch = len(dataloader)
    for ib, data in enumerate(dataloader):
        num_data =  data['videos'].shape[0]

        if SAVE_GT is True:
            for i in range(num_data):
                np.save(os.path.join(gt_save_dir, f"rgbs{gt_data_id}.npy"), data['videos'][i].cpu().numpy())
                np.save(os.path.join(gt_save_dir, f"depths{gt_data_id}.npy"), data['gt_depths'][i].cpu().numpy())            # (f, h, w)
                np.save(os.path.join(gt_save_dir, f"extrinsics{gt_data_id}.npy"), data['gt_extrinsics'][i].cpu().numpy())    # (f, 4, 4)
                np.save(os.path.join(gt_save_dir, f"intrinsics{gt_data_id}.npy"), data['gt_intrinsics'][i].cpu().numpy())    # (f, 3, 3)
                np.save(os.path.join(gt_save_dir, f"hoi_mask{gt_data_id}.npy"), data['hoi_masks'][i].cpu().numpy())          # (f, h, w)
                gt_data_id = gt_data_id + 1

        st_time = time.time()
        put_data_device(data, DEVICE)
        result = model_inference_conductor(data, model_wrapper, num_frames, step_overlap, vis=False)

        for i in range(num_data):
            np.save(os.path.join(egomono4d_save_dir, f"pcd{data_id}.npy"), result['xyzs'][i].cpu().numpy())
            np.save(os.path.join(egomono4d_save_dir, f"intrinsic{data_id}.npy"), result['intrinsics'][i].cpu().numpy())
            np.save(os.path.join(egomono4d_save_dir, f"extrinsic{data_id}.npy"), result['extrinsics'][i].cpu().numpy())
            np.save(os.path.join(egomono4d_save_dir, f"mask{data_id}.npy"), result['weights'][i].cpu().numpy())
            np.save(os.path.join(egomono4d_save_dir, f"depth{data_id}.npy"), result['depths'][i].cpu().numpy())
            data_id = data_id + 1

        epoch_time = time.time() - st_time
        rest_t = np.round(epoch_time*(n_batch-ib-1), 1)
        rest_min = rest_t // 60
        rest_sec = np.round(rest_t % 60, 1)
        rest_hr = rest_min // 60
        rest_min = rest_min % 60
        print(f"Finish Batch [{ib+1}/{n_batch}]. [{epoch_time} secs] [restTime: {rest_hr} hr {rest_min} min {rest_sec} secs]")
        
    gt_dir = os.path.join(base_save_dir, "gt", dataset_name)
    model_dir = egomono4d_save_dir
    assert len(os.listdir(gt_dir)) % 5 == 0
    n_data = len(os.listdir(gt_dir)) // 5
    eval_metric_list = []

    for i in tqdm(range(n_data), desc="EVAL"):
        
        commit = dataset_name + "_" + model_name + "_" + str(i)

        videos = torch.Tensor(np.load(os.path.join(gt_dir, f"rgbs{i}.npy")))[None].to(DEVICE)
        eval_metric = {"eval": f"long_videos_{dataset_name}_{model_name}", "num_frames": videos.shape[1],}
        gt_depths = torch.Tensor(np.load(os.path.join(gt_dir, f"depths{i}.npy")))[None].to(DEVICE)
        gt_mask = gt_depths > 0
        gt_extrinsic = torch.Tensor(np.load(os.path.join(gt_dir, f"extrinsics{i}.npy")))[None].to(DEVICE)
        gt_intrinsic = torch.Tensor(np.load(os.path.join(gt_dir, f"intrinsics{i}.npy")))[None].to(DEVICE)
        gt_hoi_masks = torch.Tensor(np.load(os.path.join(gt_dir, f"hoi_mask{i}.npy")))[None].to(DEVICE)
        gt_pcds = batch_recover_pointclouds_sequence(gt_depths+EPS*gt_mask, gt_intrinsic, gt_extrinsic)

        ###################################### Raw Prediction ##########################################
        ######## Extrinsic ########
        pred_extrinsic = torch.Tensor(np.load(os.path.join(model_dir, f"extrinsic{i}.npy")))[None].to(DEVICE)
        metric_extrinsic = eval_extrinsic_conductor(pred_extrinsic, gt_extrinsic)
        eval_metric['extrinsics'] = metric_extrinsic
        
        ######## Depths ########
        pred_depths = torch.Tensor(np.load(os.path.join(model_dir, f"depth{i}.npy")))[None].to(DEVICE)
        fly_masks = []
        for imask in range(len(pred_depths)):
            fl = detect_sequence_flying_pixels(pred_depths[imask].cpu().numpy(), threshold=FLY_THRESHOLD)
            fly_masks.append(fl)
        pred_mask = torch.Tensor(np.stack(fly_masks)).to(DEVICE)
        flys = torch.minimum(1.0 - pred_mask, gt_mask)
        metric_depth = eval_depth_conductor(pred_depths, gt_depths, flys)
        eval_metric['depths'] = metric_depth

        ######## PointCloud and Track ########
        pred_pcds = torch.Tensor(np.load(os.path.join(model_dir, f"pcd{i}.npy")))[None].to(DEVICE)
        metric_pointcloud = eval_pointcloud_conductor(pred_pcds, gt_pcds, flys, rgbs=videos, commit=commit)
        eval_metric['pointclouds'] = metric_pointcloud
        metric_track = eval_track_conductor(pred_pcds, gt_pcds, flys, rgbs=videos, tracker=tracker, commit=commit)
        eval_metric['tracks'] = metric_track
        metric_track = eval_track_hoi_conductor(pred_pcds, gt_pcds, flys, hoi_masks=gt_hoi_masks, rgbs=videos, tracker=tracker, commit=commit)
        eval_metric['tracks_hoi'] = metric_track

        eval_metric_list.append(eval_metric)


    def get_final_metric(eval_list):
        final_metric = dict()
        final_metric['eval'] = eval_list[0]['eval']
        final_metric['num_frames'] = eval_list[0]['num_frames']
        final_metric['depths'], final_metric['extrinsics'], final_metric['pointclouds'], final_metric['tracks'], final_metric['tracks_hoi'] = dict(), dict(), dict(), dict(), dict()
        final_metric_num = {'depths': dict(), "extrinsics": dict(), "pointclouds": dict(), "tracks": dict(), "tracks_hoi": dict()}
        for metric_type in ['depths', 'extrinsics', 'pointclouds', 'tracks', 'tracks_hoi']:
            for eval_item in eval_list:
                eval_res = eval_item[metric_type]
                if len(final_metric[metric_type].keys()) == 0:
                    for k in eval_item[metric_type].keys():
                        if eval_item[metric_type][k] is None:
                            final_metric[metric_type][k] = 0
                            final_metric_num[metric_type][k] = 0
                        else:
                            final_metric[metric_type][k] = eval_item[metric_type][k]
                            final_metric_num[metric_type][k] = 1
                else:
                    for k in eval_item[metric_type].keys():
                        if eval_item[metric_type][k] is None:
                            continue
                        else:
                            final_metric[metric_type][k] = final_metric[metric_type][k] + eval_item[metric_type][k]
                            final_metric_num[metric_type][k] = final_metric_num[metric_type][k] + 1
            for k in eval_item[metric_type].keys():
                final_metric[metric_type][k] = final_metric[metric_type][k] / final_metric_num[metric_type][k]
        return final_metric"

    save_base = base_save_dir
    final_eval_metric = get_final_metric(eval_metric_list)
    with open(save_base+"/"f"eval_{commit}.json", "w") as f:
        json.dump(final_eval_metric, f, indent=4)
    # # pdb.set_trace()



@torch.no_grad()
def inference(base_save_dir, args, cfg, model_name):
    
    if args.windows_size is not None:
        cfg.preprocess.num_frames = args.windows_size
    
    loss_name_list = [cfg_item.name for cfg_item in cfg.loss]
    dataset_name = cfg.dataset[0].name
    cfg.model.backbone.cache_dir = cfg.base_cache_dir
    cfg.flow.cache_dir = cfg.base_cache_dir
    cfg.tracking.cache_dir = cfg.base_cache_dir
    for dataset_cfg in cfg.dataset:
        dataset_cfg.resize_shape = cfg.preprocess.resize_shape
        dataset_cfg.patch_size = cfg.preprocess.patch_size
        dataset_cfg.num_frames = cfg.preprocess.num_frames
        dataset_cfg.cache_dir = cfg.base_cache_dir
        dataset_cfg.use_consistency_loss = ('cc' in loss_name_list)
        if hasattr(dataset_cfg, "mask_flow_model"):
            dataset_cfg.mask_flow_model = cfg.flow 
        dataset_cfg.all_frames = True
    
    model = Model(cfg.model, inference_trunc=args.inference_trunc)
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

    dataloader = DataLoader(get_dataset(cfg.dataset, "test", global_rank=0, world_size=1), 
                            batch_size=args.batch_size, shuffle=False)
    
    model_result_evaluate(base_save_dir, dataset_name, model_name, args, cfg, model_wrapper, dataloader, tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference of 4D Dynamic Scene Reconstruction.")
    parser.add_argument("--base_save_dir", type=int, default=".", help="The directory which saves the evaluation result.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="how many step of interval for trajectory consistency calculation.")
    parser.add_argument("-s", "--step_overlap", type=int, default=1, help="how many step of interval for trajectory consistency calculation.")
    parser.add_argument("-w", "--windows_size", type=int, default=None, help="the windows size used for inference.")
    parser.add_argument("-a", "--all_frames", action="store_true", help="test with all-frame (long-video) mode.")
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-f", "--folder_path", type=str, default=None)
    parser.add_argument("-t", "--inference_trunc", type=float, default=None)
    args = parser.parse_args()

    if args.folder_path is not None:
        model_name = args.folder_path.split('/')[-1]
        cfg = get_cfg(args)
        args.model = args.folder_path 
        flist = os.listdir(args.model)
        for fp in flist:
            if fp.startswith("train_ddp"):
                continue
            if fp == ".hydra" or fp.endswith(".log"): 
                continue
            args.model = args.model + "/" + fp + "/egomono4d"
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
    inference(args.base_save_dir, args, cfg, model_name)


# CUDA_VISIBLE_DEVICES=0 python -m egomono4d.evaluation -f egomono4d_model -b 8 -c /root/project/dflowmap/config/pretrain_eval_hoi4d.yaml
# CUDA_VISIBLE_DEVICES=0 python -m egomono4d.evaluation -f egomono4d_model -b 8 -c /root/project/dflowmap/config/pretrain_eval_h2o.yaml
# CUDA_VISIBLE_DEVICES=0 python -m egomono4d.evaluation -f egomono4d_model -b 8 -c /root/project/dflowmap/config/pretrain_eval_arctic.yaml
# CUDA_VISIBLE_DEVICES=0 python -m egomono4d.evaluation -f egomono4d_model -b 8 -c /root/project/dflowmap/config/pretrain_eval_pov_surgery.yaml