from dataclasses import dataclass
import os
import re
import pdb
import json
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
from ..misc.fly import detect_sequence_flying_pixels

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    from .mpreprocess_images import parallel_preprocess
    from ..misc.data_util import resize_crop_intrinisic, compute_patch_cropped_shape, canonicalize_intrinisic


FLY_THRESHOLD = 0.05

@dataclass
class DatasetH2OCfg(DatasetCfgCommon):
    name: Literal["h2o"] = "h2o"
    # name: str = "h2o"
    clip_frame: int = None

    original_base_root: str = None
    pre_save_root: str = None


class DatasetH2O(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetH2OCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)

        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_h2o_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For HOI4D dataset, you must appoint a pre_save_root for preprocessing (not None).")
        self.pre_save_root = pre_save_root

        os.makedirs(self.pre_save_root, exist_ok=True)
        print(f"preprocess_dir = {self.pre_save_root}")
        print(f"len = {len(os.listdir(self.pre_save_root))}")
        if len(os.listdir(self.pre_save_root)) == 0:
            self._get_preprocess_save(cfg.original_base_root)

        with open(os.path.join(self.pre_save_root, f"sequence_{stage}.json"), "r") as f:
            self.sequence = json.load(f)


    def _get_preprocess_save(self, root_dir):
        
        os.makedirs(self.pre_save_root, exist_ok=True)
        save_data_dir = os.path.join(self.pre_save_root, "h2o_preprocessed")
        os.makedirs(save_data_dir, exist_ok=True)

        ########################### Get Original Sequence ##########################

        def extract_number(filename):
            filename = filename.split('/')[-1]
            match = re.search(r'\d+', filename)  
            return int(match.group()) if match else 0  
        
        sequence_original = []
        for P in tqdm(["subject1", "subject2", "subject3", "subject4"], desc="Subject"):
            P_dir = os.path.join(root_dir, P+"_ego")
            PV_list = os.listdir(P_dir)
            os.makedirs(os.path.join(save_data_dir, P), exist_ok=True)
            for V in tqdm(PV_list, desc="V"): 
                V_dir = os.path.join(P_dir, V)          
                os.makedirs(os.path.join(save_data_dir, P, V), exist_ok=True)
                PVR_list = os.listdir(V_dir)
                for R in tqdm(PVR_list, desc="R"):   # subject1_ego/h1/0
                    R_dir = os.path.join(V_dir, R, "cam4")
                    rgb_list = os.listdir(os.path.join(R_dir, "rgb"))  
                    dep_list = os.listdir(os.path.join(R_dir, "depth"))  
                    extr_list = os.listdir(os.path.join(R_dir, "cam_pose")) 
                    rgb_list = [os.path.join(R_dir, "rgb", x) for x  in rgb_list]
                    dep_list = [os.path.join(R_dir, "depth", x) for x  in dep_list]
                    extr_list = [os.path.join(R_dir, "cam_pose", x) for x  in extr_list]
                    intr = os.path.join(R_dir, "cam_intrinsics.txt")
                    rgb_list.sort(key=extract_number)
                    dep_list.sort(key=extract_number)
                    extr_list.sort(key=extract_number)
                    data = {
                        "id": f"H2O_{P}_{V}_{R}",
                        "P": P, "V": V,
                        "rgbs": rgb_list,
                        "deps": dep_list, 
                        "extrs": extr_list,
                        "intr": intr
                    }
                    sequence_original.append(data)

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        # #################### Preprare Sequential List for Parallel Preprocess ####################
        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "r") as f:
            sequence_original = json.load(f)
        
        sequence_preprocess = []
        for seq in sequence_original:
            uid_path = os.path.join(save_data_dir, seq['P'], seq['V'])
            rgbs_list = seq['rgbs']
            deps_list = seq['deps']
            extrs_list = seq['extrs']
            rgbs_list = rgbs_list[::2]
            deps_list = deps_list[::2]
            extrs_list = extrs_list[::2]
            n_images = len(rgbs_list)
            n_seq_data = 0
            assert n_images == len(deps_list)

            seq_l = 0
            while seq_l + self.cfg.clip_frame <= n_images:
                seq_r = seq_l + self.cfg.clip_frame
                data_dict = {"id": seq['id']+"_"+str(n_seq_data), "save_dir": os.path.join(uid_path, str(n_seq_data)), "intr": seq['intr'], "base_id": 0}
                rgbs_clip, deps_clip, extrs_clip = [], [], []
                for i in range(seq_l, seq_r):
                    rgbs_clip.append(seq['rgbs'][i])
                    deps_clip.append(seq['deps'][i])
                    extrs_clip.append(seq['extrs'][i])
                data_dict['rgbs'] = rgbs_clip
                data_dict['deps'] = deps_clip
                data_dict['extrs'] = extrs_clip
                seq_l = seq_l + self.cfg.clip_frame // 2
                sequence_preprocess.append(data_dict)
                n_seq_data = n_seq_data + 1
            print(f"P: {seq['P']} ; V:{seq['V']} ; n_seq_data={n_seq_data}")
        print(f"len_sequence={len(sequence_preprocess)}")

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "w") as f:
            json.dump(sequence_preprocess, f, indent=4)

        ########################### Conduct Parallel Preprocess ###########################

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "r") as f:
            sequence_preprocess = json.load(f)

        sequence = parallel_preprocess(
            n_cpu_procs=64,
            n_gpu_procs=8,
            num_gpu=torch.cuda.device_count(),
            cfg=self.cfg,
            sequence=sequence_preprocess,
            pre_save_root=self.pre_save_root
        )

        with open(os.path.join(self.pre_save_root, f"sequence_processed.json"), "r") as f:
            sequence = json.load(f)

        resize_shape = self.cfg.resize_shape
        w_old, h_old = Image.open(sequence_preprocess[0]['rgbs'][0]).size
        h_new, w_new = resize_shape
        scale_factor = max(h_new / h_old, w_new / w_old)
        h_scaled = round(h_old * scale_factor)
        w_scaled = round(w_old * scale_factor) 
        patch_size = self.cfg.patch_size
        patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)
        h_pt, w_pt = patch_crop_shape

        sequence_processed_camera = []
        for seq in tqdm(sequence):
            extrs_org = seq['extrs']
            rgbs = seq['rgbs']
            intr_fp = seq['intr']
            base_dir = '/'.join(rgbs[0].split('/')[:-1])
            with open(intr_fp, 'r') as file:
                data = file.readline().split()
            fx, fy, cx, cy, width, height = map(float, data)
            intrinsic_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

            intr_new = resize_crop_intrinisic(torch.tensor(intrinsic_matrix), (h_old, w_old), (h_scaled, w_scaled), (h_new, w_new))
            intr_c = canonicalize_intrinisic(intr_new, (h_pt, w_pt)).cpu().numpy()
            np.save(base_dir+"/"+"intrinsic_canonical.npy", intr_c)

            extr_list = []
            for extr_fp in extrs_org:
                with open(extr_fp, 'r') as file:
                    data = file.readline().split()
                data = list(map(float, data))
                extrinsic_matrix = np.array(data).reshape((4, 4))
                extr_list.append(extrinsic_matrix)
            
            extr_new = np.stack(extr_list)
            np.save(base_dir+"/"+"extrinsic.npy", extr_new)

            seq['intr'] = base_dir+"/"+"intrinsic_canonical.npy"
            seq['extr'] = base_dir+"/"+"extrinsic.npy"
            sequence_processed_camera.append(seq)

        with open(os.path.join(self.pre_save_root, "sequence_processed_camera.json"), "w") as f:
            json.dump(sequence_processed_camera, f, indent=4)
        
        sequence = sequence_processed_camera

        train_id = ['subject1_h1', 'subject1_h2', 'subject1_k1', 'subject1_k2', 'subject1_o1', 'subject1_o2', 'subject2_h1', 'subject2_h2', 'subject2_k1', 'subject2_k2', 'subject2_o1', 'subject2_o2', 'subject3_h1', 'subject3_h2', 'subject3_k1']
        val_id = ['subject3_k2', 'subject3_o1', 'subject3_o2']
        test_id = ['subject4_h1', 'subject4_h2', 'subject4_k1', 'subject4_k2', 'subject4_o1', 'subject4_o2']
        sequence_train = [x for x in sequence if '_'.join(x['id'].split('_')[1:3]) in train_id]
        sequence_val = [x for x in sequence if '_'.join(x['id'].split('_')[1:3]) in val_id]
        sequence_test = [x for x in sequence if '_'.join(x['id'].split('_')[1:3]) in test_id]

        with open(os.path.join(self.pre_save_root, "sequence_train.json"), "w") as f:
            json.dump(sequence_train, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_val.json"), "w") as f:
            json.dump(sequence_val, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_test.json"), "w") as f:
            json.dump(sequence_test, f, indent=4)

        print(f"len(train)={len(sequence_train)} ; len(val)={len(sequence_val)} ; len(test)={len(sequence_test)}")
        print(f"len_all={len(sequence)}")

    
    def get_item_wrapper_single(self, index: int):
        
        meta = self.sequence[index]
        num_frames = len(meta["rgbs"])
        if (self.all_frames is True) and (self.stage in ['val', 'test']):
            indices = torch.Tensor([i for i in range(num_frames)], device=torch.device('cpu')).int()
        else:
            indices = self.frame_sampler.sample(num_frames, torch.device("cpu"), self.frame_max_interval)
        rgbs_fp = [meta['rgbs'][i] for i in indices]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgbs_fp])
        deps_fp = []
        intr_fp = []
        for rgb_fp in rgbs_fp:
            aux_fp = self.get_aux_filename(rgb_fp)
            deps_fp.append(aux_fp['edep'])
            intr_fp.append(aux_fp['intrinsics'])

        if self.cfg.use_gt_depth is False:
            depths = torch.stack([torch.Tensor(np.load(path)) for path in deps_fp])
            intrinsics = torch.stack([torch.Tensor(np.load(path)) for path in intr_fp])
            fly_mask = detect_sequence_flying_pixels(depths.numpy(), threshold=FLY_THRESHOLD)
            fly_mask = 1.0 - torch.Tensor(fly_mask)
        else:
            deps_fp = [meta['deps'][i] for i in indices]
            depths = torch.stack([tf.ToTensor()(Image.open(path)) for path in deps_fp]).squeeze(1)
            fly_mask = 1.0 - (depths == 0).float()
            depths = depths / 1000.0
            intrinsics = torch.stack([torch.Tensor(np.load(meta['intr']))]*videos.shape[0])

        if (EVAL in ['True']):
            masks = torch.zeros_like(depths)
        else:
            masks_fp = [meta['emasks'][i] for i in indices]
            masks = []
            for fp in masks_fp:
                msk = np.load(fp)
                if msk.ndim == 3: msk = msk.sum(2)
                masks.append(msk)
            masks = torch.Tensor(np.stack(masks))

        data = {
            "videos": videos,      # (F, C, H, W)
            "depths": depths,    
            "flys": fly_mask,
            "masks": torch.minimum(masks, fly_mask), 
            "indices": indices,
            "scenes": meta['id'],
            "datasets": "h2o",
            "intrinsics": intrinsics,
            "use_gt_depth": self.cfg.use_gt_depth,
            "hoi_masks": 1.0 - masks
        }

        # H2O test setting
        if (self.stage in ['test']) or (self.cfg.use_gt_depth is True):
            deps_fp = [meta['deps'][i] for i in indices]
            gt_depths = torch.stack([tf.ToTensor()(Image.open(path)) for path in deps_fp]).squeeze(1)
            data['gt_depths'] = gt_depths / 1000.0
            f = videos.shape[0]
            data['gt_intrinsics'] = torch.stack([torch.Tensor(np.load(meta['intr']))]*f)
            data['gt_extrinsics'] = torch.Tensor(np.load(meta['extr']))[indices] 

        return data
