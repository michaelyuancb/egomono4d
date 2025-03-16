from dataclasses import dataclass
import os
import re
import pdb
import json
import numpy as np
from tqdm import tqdm
from typing_extensions import Literal

import torch
import torchvision.transforms as tf
from PIL import Image

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon, Datasetegomono4d
from .types import Stage


try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    from ..misc.depth import get_depth_estimator, estimate_relative_depth
    from ..misc.data_util import pil_resize_to_center_crop, resize_crop_intrinisic, compute_patch_cropped_shape, canonicalize_intrinisic

FLY_THRESHOLD = 0.05
EVAL_VIDEO = [
    "box_grab_01", "box_use_01", "capsulemachine_grab_01", "capsulemachine_use_01",
    "espressomachine_grab_01", "espressomachine_use_01", "ketchup_grab_01",
    "ketchup_use_01", "laptop_grab_01", "laptop_use_01", "microwave_grab_01",
    "microwave_use_01", "mixer_grab_01", "mixer_use_01",
    "notebook_use_01", "phone_grab_01", "phone_use_01", 
    "scissors_grab_01", "scissors_use_01", "waffleiron_grab_01", "waffleiron_use_01"
]


@dataclass
class DatasetArcticCfg(DatasetCfgCommon):
    name: Literal["arctic"] = "arctic"
    # name: str = "arctic"
    clip_frame: int = None

    original_data_root: str = None
    original_render_root: str = None
    pre_save_root: str = None


class DatasetArctic(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetArcticCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)

        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_arctic_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For ARCTIC dataset, you must appoint a pre_save_root for preprocessing (not None).")
        self.pre_save_root = pre_save_root

        os.makedirs(self.pre_save_root, exist_ok=True)
        print(f"preprocess_dir = {self.pre_save_root}")
        print(f"len = {len(os.listdir(self.pre_save_root))}")
        if len(os.listdir(self.pre_save_root)) == 0:
            self._get_preprocess_save(cfg.original_data_root, cfg.original_render_root)
        # self._get_preprocess_save(cfg.original_data_root, cfg.original_render_root)

        with open(os.path.join(self.pre_save_root, f"sequence_datapoint.json"), "r") as f:
            self.sequence = json.load(f)
        
        if self.stage in ['train']:
            self.sequence = self.sequence * 45
        elif self.stage in ['val']:
            self.sequence = self.sequence * 2


    def _get_preprocess_save(self, data_dir, render_dir):
        
        os.makedirs(self.pre_save_root, exist_ok=True)
        save_data_dir = os.path.join(self.pre_save_root, "arctic_preprocessed")
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
        
        sequence_original = []
        for iA, A in tqdm(enumerate(EVAL_VIDEO), desc="Action"):
            
            print(f"generate {A}.")

            A_rgb_dir = os.path.join(data_dir, "arctic_data/data/images/s01", A, "0")
            A_cam_file = os.path.join(data_dir, "arctic_data/data/raw_seqs/s01", A+".egocam.dist.npy")
            A_depth_dir = os.path.join(render_dir, "s01_"+A+"_0", 'images/depth')
            A_mask_dir = os.path.join(render_dir, "s01_"+A+"_0", 'images/mask')

            dep_list = os.listdir(A_depth_dir)
            nd = len(dep_list)
            rgb_list = os.listdir(A_rgb_dir)[:nd]
            mask_list = os.listdir(A_mask_dir) 
            rgb_list.sort(key=extract_number)
            dep_list.sort(key=extract_number)
            mask_list.sort(key=extract_number)

            cam_info = np.load(A_cam_file, allow_pickle=True).item()
            intrinsic = np.array(cam_info['intrinsics'])                 # [3, 3]
            R_extrinsic = cam_info['R_k_cam_np']                         # [T, 3, 3]
            T_extrinsic = cam_info['T_k_cam_np']                         # [T, 3, 1]
            extrinsics = np.concatenate([R_extrinsic, T_extrinsic], axis=-1)
            extrinsics = np.concatenate([extrinsics, np.zeros((R_extrinsic.shape[0], 1, 4))], axis=1)
            extrinsics[:, -1, -1] = 1.0

            n_images_org = len(rgb_list)
            assert n_images_org == len(dep_list)
            assert n_images_org == len(mask_list)
            assert n_images_org == R_extrinsic.shape[0]
            assert n_images_org == T_extrinsic.shape[0]
            if n_images_org < 20: continue

            rgb_list, dep_list, mask_list = rgb_list[10:n_images_org-10], dep_list[10:n_images_org-10], mask_list[10:n_images_org-10]
            extrinsics = extrinsics[10:n_images_org-10]

            n_images = len(rgb_list)
            A_save_dir = os.path.join(save_data_dir, A)
            os.makedirs(A_save_dir, exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "rgbs"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "deps"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "edeps"), exist_ok=True)
            os.makedirs(os.path.join(A_save_dir, "masks"), exist_ok=True)
            np.save(A_save_dir+"/"+"extrinsics.npy", extrinsics)

            meta = {"id": "ARCTIC_"+A, "rgbs":[], "deps": [], "masks": [], "extrinsics": A_save_dir+"/"+"extrinsics.npy"}
            meta['edeps'] = []
            meta['edeps_intrinsic'] = []

            w_old, h_old = Image.open(os.path.join(A_rgb_dir, rgb_list[0])).size
            for i in tqdm(range(n_images), desc='Frames'):
                rgb = Image.open(os.path.join(A_rgb_dir, rgb_list[i]))
                rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, resize_shape, patch_crop_shape)
                rgb_fp = os.path.join(A_save_dir, "rgbs", "rgb"+str(i)+".png")
                rgb.save(rgb_fp)
                meta['rgbs'].append(rgb_fp)

                dep = np.load(os.path.join(A_depth_dir, dep_list[i]))[:, 170:1430]  
                assert dep.shape[0] / dep.shape[1] == h_old / w_old
                dep[dep > 8.0] = 0.0
                dep = Image.fromarray((dep*10000).astype(np.uint32))
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

                mask = Image.open(os.path.join(A_mask_dir, mask_list[i]))
                mask = Image.fromarray(np.array(mask)[:, 170:1430])
                assert mask.size[1] / mask.size[0] == h_old / w_old
                mask, (h_scaled, w_scaled) = pil_resize_to_center_crop(mask, resize_shape, patch_crop_shape)
                mask_fp = os.path.join(A_save_dir, "masks", "mask"+str(i)+".png")
                mask.save(mask_fp)
                meta['masks'].append(mask_fp)

            intr_new = resize_crop_intrinisic(torch.tensor(intrinsic), (h_old, w_old), (h_scaled, w_scaled), (h_new, w_new))
            np.save(A_save_dir+"/"+"intrinsic.npy", intr_new)
            intr_c = canonicalize_intrinisic(intr_new, (h_pt, w_pt)).cpu().numpy()
            np.save(A_save_dir+"/"+"intrinsic_canonical.npy", intr_c)
            meta['intrinsic'] = A_save_dir+"/"+"intrinsic_canonical.npy"

            sequence_original.append(meta)

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        # #################### Preprare Sequential List for Parallel Preprocess ####################
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
            masks_list = meta_org['masks']
            intrinsic_fp = meta_org['intrinsic']
            extrinsics_fp = meta_org['extrinsics']
            extrinsics = np.load(extrinsics_fp)
            n_images = len(rgbs_list)

            seq_l = 0
            n_seq_data = 0
            while seq_l + self.cfg.clip_frame <= n_images:
                seq_r = seq_l + self.cfg.clip_frame
                exts_fp = os.path.join(A_save_dir, "exts", f"exts{n_seq_data}.npy")
                np.save(exts_fp, extrinsics[seq_l:seq_r])

                data = {
                    "id": id_org + "_" + str(n_seq_data) + f"_({seq_l},{seq_r})",
                    "action": A,
                    "rgbs": rgbs_list[seq_l:seq_r],
                    "deps": deps_list[seq_l:seq_r],
                    "edeps": edeps_list[seq_l:seq_r],
                    "masks": masks_list[seq_l:seq_r],
                    "intrinsic": intrinsic_fp,
                    "extrinsics": str(exts_fp)
                }

                sequence_preprocess.append(data)
                seq_l = seq_l + self.cfg.clip_frame
                n_seq_data = n_seq_data + 1

        print(f"len_sequence={len(sequence_preprocess)}")

        with open(os.path.join(self.pre_save_root, "sequence_datapoint.json"), "w") as f:
            json.dump(sequence_preprocess, f, indent=4)


    def get_item_wrapper_single(self, index):
        meta = self.sequence[index]
        num_frames = len(meta["rgbs"])
        if (self.all_frames is True) and (self.stage in ['val', 'test']):
            indices = torch.Tensor([i for i in range(num_frames)], device=torch.device('cpu')).int()
        else:
            indices = self.frame_sampler.sample(num_frames, torch.device("cpu"), self.frame_max_interval)
        
        rgbs_fp = [meta['rgbs'][i] for i in indices]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgbs_fp])
        if (EVAL in ['True']):
            depths = torch.zeros((videos.shape[0], videos.shape[2], videos.shape[3]))
        else:
            deps_fp = [meta['edeps'][i] for i in indices]
            depths = torch.stack([tf.ToTensor()(np.load(path)) for path in deps_fp])[:, 0]
        masks_fp = [meta['masks'][i] for i in indices]
        masks = torch.stack([tf.ToTensor()(Image.open(path)) for path in masks_fp])
        masks = 1.0 - (masks.sum(dim=1) > 0).float()
        (f, c, h, w) = videos.shape


        data = {
            "videos": videos,      # (F, C, H, W)
            "depths": depths,    
            "flys": 1.0 - masks,
            "masks": masks,
            "indices": indices,
            "scenes": meta['id'],
            "datasets": "arctic",
            "use_gt_depth": True,
            "hoi_masks": 1.0 - masks,
        }

        gt_deps_fp = [meta['deps'][i] for i in indices]
        data['gt_depths'] = torch.stack([torch.Tensor(np.load(path)) for path in gt_deps_fp])
        f = videos.shape[0]
        data['gt_intrinsics'] = torch.stack([torch.Tensor(np.load(meta['intrinsic']))]*f)
        data['gt_extrinsics'] = torch.Tensor(np.load(meta['extrinsics']))[indices].inverse()
        data['intrinsics'] = data['gt_intrinsics']
        data['depths'] = data['gt_depths']

        return data