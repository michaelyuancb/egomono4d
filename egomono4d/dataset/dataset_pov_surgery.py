from dataclasses import dataclass
import os
import re
import pdb
import json
import pickle
import numpy as np
from tqdm import tqdm
# from typing import Literal
from typing_extensions import Literal

import torch
import torchvision.transforms as tf
from PIL import Image

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon, Datasetegomono4d
from .types import Stage
from ..misc.data_util import pil_resize_to_center_crop, resize_crop_intrinisic, compute_patch_cropped_shape, canonicalize_intrinisic

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
try:
    INFER = os.environ['INFER_MODE']
except:
    INFER = 'False'
if (EVAL not in ['True']) and (INFER not in ['True']):
        from ..hoi.ego_hos_wrapper import EgoHOSWrapper
        from ..misc.depth import get_depth_estimator, estimate_relative_depth
else:
    from ..misc.depth import get_depth_estimator, estimate_relative_depth


FLY_THRESHOLD = 0.05


@dataclass
class DatasetPOVSurgeryCfg(DatasetCfgCommon):
    name: Literal["pov_surgery"] = "pov_surgery"
    # name: str = "pov_surgery"
    clip_frame: int = None

    original_base_root: str = None
    pre_save_root: str = None


class DatasetPOVSurgery(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetPOVSurgeryCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)

        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_pov_surgery_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For HOI4D dataset, you must appoint a pre_save_root for preprocessing (not None).")
        self.pre_save_root = pre_save_root

        os.makedirs(self.pre_save_root, exist_ok=True)
        print(f"preprocess_dir = {self.pre_save_root}")
        print(f"len = {len(os.listdir(self.pre_save_root))}")
        if len(os.listdir(self.pre_save_root)) == 0:
            self._get_preprocess_save(cfg.original_base_root)
        # self._get_preprocess_save(cfg.original_base_root)

        with open(os.path.join(self.pre_save_root, f"sequence_datapoint.json"), "r") as f:
            self.sequence = json.load(f)

        gt_intrinsic_org = torch.Tensor([[1198.4395, 0, 960], [0, 1198.4395, 175.2], [0, 0, 1.0]])  
        h_old, w_old = 1080.0, 1920.0
        resize_shape = self.cfg.resize_shape
        h_new, w_new = resize_shape
        scale_factor = max(h_new / h_old, w_new / w_old)
        h_scaled = round(h_old * scale_factor)
        w_scaled = round(w_old * scale_factor) 
        patch_size = self.cfg.patch_size
        patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)
        h_pt, w_pt = patch_crop_shape
        intr_new = resize_crop_intrinisic(gt_intrinsic_org, (h_old, w_old), (h_scaled, w_scaled), (h_new, w_new))
        intr_c = canonicalize_intrinisic(intr_new, (h_pt, w_pt))
        self.gt_intrinsic = intr_c
        self.opengl2opencv = torch.Tensor([[1,  0,  0, 0], [0, -1,  0, 0], [0,  0, -1, 0], [0,  0,  0, 1]])

    def _get_preprocess_save(self, original_base_root):
        
        os.makedirs(self.pre_save_root, exist_ok=True)
        save_data_dir = os.path.join(self.pre_save_root, "pov_surgery_preprocessed")
        os.makedirs(save_data_dir, exist_ok=True)

        ########################### Get Original Sequence ##########################

        def extract_number(filename):
            filename = filename.split('/')[-1]
            match = re.search(r'\d+', filename)  
            return int(match.group()) if match else 0  

        resize_shape = self.cfg.resize_shape
        h_new, w_new = resize_shape
        patch_size = self.cfg.patch_size
        patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)
        h_pt, w_pt = patch_crop_shape
        depth_estimator = get_depth_estimator(cache_dir=self.cfg.cache_dir, device='cuda')
        os.makedirs(f"{self.cfg.cache_dir}/ego_hos_cache", exist_ok=True)
        mask_egohos_model = EgoHOSWrapper(cache_path=os.path.abspath(f"{self.cfg.cache_dir}/ego_hos_cache"), 
                                          repo_path=f"{self.cfg.cache_dir}/ego_hos_checkpoints", device='cuda')

        sequence_original = []
        action_list = os.listdir(os.path.join(original_base_root, "color"))
        for iA, A in tqdm(enumerate(action_list), desc="Action"):

            if not A.endswith("_1"):       # we only choose the first record of each action
                continue
            
            print(f"generate {A}.")

            A_rgb_dir = os.path.join(original_base_root, "color", A)
            A_dep_dir = os.path.join(original_base_root, "depth", A)
            A_cam_dir = os.path.join(original_base_root, "annotation", A)

            rgb_list = os.listdir(A_rgb_dir)
            dep_list = os.listdir(A_dep_dir)
            cam_list = os.listdir(A_cam_dir)
            rgb_list.sort(key=extract_number)
            dep_list.sort(key=extract_number)
            cam_list.sort(key=extract_number)
            
            n_images_org = len(cam_list)
            if cam_list[0] != '00001.pkl': continue
            if rgb_list[0] == '00000.jpg':
                rgb_list = rgb_list[1:]
            if dep_list[0] == '00000.png':
                dep_list = dep_list[1:]

            if n_images_org != len(dep_list): continue
            if n_images_org != len(cam_list): continue
            if n_images_org < 840: continue
            rgb_list, dep_list, cam_list = rgb_list[400:n_images_org-400], dep_list[400:n_images_org-400], cam_list[400:n_images_org-400]

            n_images = len(rgb_list)
            A_save_dir = os.path.join(save_data_dir, A)
            os.makedirs(A_save_dir, exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "rgbs"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "deps"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "edeps"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "emasks"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "extrinsics"), exist_ok=True)

            meta = {"id": "POV_Surgery_"+A, "rgbs":[], "deps": [], "emasks": [], "extrinsics": [], "edeps": [], "edeps_intrinsic": []}

            w_old, h_old = Image.open(os.path.join(A_rgb_dir, rgb_list[0])).size
            for i in tqdm(range(n_images), desc='Frames'):
                rgb = Image.open(os.path.join(A_rgb_dir, rgb_list[i]))
                rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, resize_shape, patch_crop_shape)
                rgb_fp = os.path.join(A_save_dir, "rgbs", "rgb"+str(i)+".png")
                rgb.save(rgb_fp)
                meta['rgbs'].append(rgb_fp)

                dep = Image.open(os.path.join(A_dep_dir, dep_list[i]))
                assert dep.size[0] / dep.size[1] == w_old / h_old
                dep, _ = pil_resize_to_center_crop(dep, resize_shape, patch_crop_shape, depth_process=True)
                dep = np.array(dep) / 10000.0
                dep_fp = os.path.join(A_save_dir, "deps", "dep"+str(i)+".npy")
                np.save(dep_fp, dep)
                meta['deps'].append(str(dep_fp))

                if depth_estimator is not None:
                    e_dep_pack = estimate_relative_depth(rgb, depth_estimator)
                    e_dep = e_dep_pack['depth']
                    e_dep_fp_np = os.path.join(A_save_dir, "edeps", "edep"+str(i)+".npy")
                    np.save(e_dep_fp_np, e_dep)
                    meta['edeps'].append(e_dep_fp_np)
                    e_dep_intr = e_dep_pack['intrinsics']
                    e_dep_intr_can = canonicalize_intrinisic(torch.tensor(e_dep_intr), (h_new, w_new))
                    e_dep_intr_np = os.path.join(A_save_dir, "edeps", "intrinsic"+str(i)+".npy")
                    np.save(e_dep_intr_np, e_dep_intr_can.cpu().numpy())
                    meta['edeps_intrinsic'].append(e_dep_intr_np)

                if mask_egohos_model is not None:
                    hand, obj, cb = mask_egohos_model.segment(rgb_fp)
                    mask = (hand + obj + cb) == 0
                    mask = mask.astype(np.float32)
                    e_mask_fp_np = os.path.join(A_save_dir, "emasks", "emasks"+str(i)+".npy")
                    np.save(e_mask_fp_np, mask)
                    meta['emasks'].append(e_mask_fp_np)

                cam_info = pickle.load(open(os.path.join(A_cam_dir, cam_list[i]), "rb"))
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = cam_info['cam_rot']
                extrinsic[:3, -1] = cam_info['cam_transl']
                e_extr_fp = os.path.join(A_save_dir, "extrinsics", "extrinsic"+str(i)+".npy")
                np.save(e_extr_fp, extrinsic)
                meta['extrinsics'].append(e_extr_fp)

            sequence_original.append(meta)

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        #################### Preprare Sequential List for Parallel Preprocess ####################
        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "r") as f:
            sequence_original = json.load(f)
        
        sequence_preprocess = []
        for meta_org in tqdm(sequence_original):
            
            id_org = meta_org['id']
            A = '_'.join(id_org.split('_')[1:])
            A_save_dir = os.path.join(save_data_dir, A)
            os.makedirs(os.path.join(A_save_dir, "exts"), exist_ok=True)
            rgbs_list = meta_org['rgbs']
            deps_list = meta_org['deps']
            edeps_list = meta_org['edeps']
            masks_list = meta_org['emasks']
            extrinsics_list = meta_org['extrinsics']
            edep_intrinsics_list = meta_org['edeps_intrinsic']
            n_images = len(rgbs_list)

            seq_l = 0
            n_seq_data = 0
            while seq_l + self.cfg.clip_frame <= n_images:
                seq_r = seq_l + self.cfg.clip_frame

                data = {
                    "id": id_org + "_" + str(n_seq_data) + f"_({seq_l},{seq_r})",
                    "action": A,
                    "rgbs": rgbs_list[seq_l:seq_r],
                    "deps": deps_list[seq_l:seq_r],
                    "edeps": edeps_list[seq_l:seq_r],
                    "masks": masks_list[seq_l:seq_r],
                    "intrinsics": edep_intrinsics_list[seq_l:seq_r],
                    "extrinsics": extrinsics_list[seq_l:seq_r]
                }

                sequence_preprocess.append(data)
                seq_l = seq_l + self.cfg.clip_frame
                n_seq_data = n_seq_data + 1

        print(f"len_sequence={len(sequence_preprocess)}")

        with open(os.path.join(self.pre_save_root, "sequence_datapoint.json"), "w") as f:
            json.dump(sequence_preprocess, f, indent=4)


    def get_item_wrapper(self, index):

        meta = self.sequence[index]
        num_frames = len(meta["rgbs"])
        if (self.all_frames is True) and (self.stage in ['val', 'test']):
            indices = torch.Tensor([i for i in range(num_frames)], device=torch.device('cpu')).int()
        else:
            indices = self.frame_sampler.sample(num_frames, torch.device("cpu"), self.frame_max_interval)
        
        rgbs_fp = [meta['rgbs'][i] for i in indices]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgbs_fp])
        (f, c, h, w) = videos.shape
        if (EVAL in ['True']):
            depths = torch.zeros((videos.shape[0], videos.shape[2], videos.shape[3]))
            masks = torch.zeros((videos.shape[0], videos.shape[2], videos.shape[3]))
            intrinsics = torch.zeros((videos.shape[0], 3, 3))
        else:
            deps_fp = [meta['edeps'][i] for i in indices]
            depths = torch.stack([torch.Tensor(np.load(path)) for path in deps_fp])
            masks_fp = [meta['masks'][i] for i in indices]
            masks = torch.stack([torch.Tensor(np.load(path)) for path in masks_fp])
            intr_fp = [meta['intrinsics'][i] for i in indices]
            intrinsics = torch.stack([torch.Tensor(np.load(path)) for path in intr_fp])

        data = {
            "videos": videos,      # (F, C, H, W)
            "depths": depths,    
            "indices": indices,
            "scenes": meta['id'],
            "datasets": "pov_surgery",
            'pcds': torch.zeros((f, h, w, 3)),
            'flys': torch.zeros((f, h, w)),
            "masks": masks,
            "intrinsics": intrinsics,
            "hoi_masks": 1.0 - masks
        }

        gt_deps_fp = [meta['deps'][i] for i in indices]
        data['gt_depths'] = torch.stack([torch.Tensor(np.load(path)) for path in gt_deps_fp]) * 2.0   # scale 5000
        f = videos.shape[0]
        data['gt_intrinsics'] = torch.stack([self.gt_intrinsic]*f)
        gt_ext_fp = [meta['deps'][i].replace('dep', 'extrinsic') for i in indices]

        data['gt_extrinsics'] = torch.stack([torch.Tensor(np.load(path)) for path in gt_ext_fp])
        data['gt_extrinsics'] = self.opengl2opencv @ data['gt_extrinsics'] @ self.opengl2opencv.T

        return data