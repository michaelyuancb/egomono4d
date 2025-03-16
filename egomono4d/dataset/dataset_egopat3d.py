from dataclasses import dataclass
import os
import re
import cv2
import pdb
import json
from tqdm import tqdm
# from typing import Literal
from typing_extensions import Literal

import torch
import time
import multiprocessing as mp

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon, Datasetegomono4d
from .types import Stage

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    from .mpreprocess_images import parallel_preprocess

FLY_THRESHOLD = 0.05
N_USE_EGOPAT3D = 6000


def process_mp4_conductor(S_dir, base_video_org_frame_dir, proc_id):
    cap = cv2.VideoCapture(os.path.join(S_dir, "rgb_video.mp4"))
    frame_count = 0
    while True:
        st = time.time()
        ret, frame = cap.read() 
        if not ret:
            break  
        frame_filename = os.path.join(base_video_org_frame_dir, f'{frame_count}.png')
        original_size = (frame.shape[1], frame.shape[0])
        target_size = (frame.shape[1] // 7, frame.shape[0] // 7)   # resize 4K video 
        resized_frame = cv2.resize(frame, target_size)
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        if frame_count >= N_USE_EGOPAT3D:     # only use N_USE_EGOPAT3D frames for each EgoPAT3D video
            break
        # if frame_count % 200 == 0:
        print(f"Proc{proc_id}: {frame_count}/{N_USE_EGOPAT3D}; rest_time: {(time.time()-st)*(N_USE_EGOPAT3D-frame_count)//60} min.")
    cap.release()
    return f"Finished {proc_id}."
                       


@dataclass
class DatasetEgoPAT3DCfg(DatasetCfgCommon):
    name: Literal["egopat3d"] = "egopat3d"
    clip_frame: int = None

    original_base_root: str = None
    pre_save_root: str = None


class DatasetEgoPAT3D(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetEgoPAT3DCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)
       
        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_egopat3d_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For EgoPAT3D dataset, you must appoint a pre_save_root for preprocessing (not None).")
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
        save_data_dir = os.path.join(self.pre_save_root, "egopat3d_preprocessed")
        os.makedirs(save_data_dir, exist_ok=True)

        ########################### Get Original Sequence ##########################

        def extract_number(filename):
            filename = filename.split('/')[-1]
            match = re.search(r'\d+', filename)  
            return int(match.group()) if match else 0  

        save_org_frame_dir = os.path.join(self.pre_save_root, "org_frame")
        os.makedirs(save_org_frame_dir, exist_ok=True)
        
        ############################### Process MP4 Video ####################################
        sequence_mp4 = []
        user_id_list = os.listdir(root_dir)
        for U in user_id_list:
            U_dir = os.path.join(root_dir, U)
            US_list = os.listdir(U_dir)
            US_list = [x for x in US_list if (('_' in x) and ('.' not in x))]   # find folder
            os.makedirs(os.path.join(save_org_frame_dir, U), exist_ok=True)
            for S in US_list:
                S_dir = os.path.join(U_dir, S)
                base_video_org_frame_dir = os.path.join(save_org_frame_dir, U, S)
                os.makedirs(base_video_org_frame_dir, exist_ok=True)
                sequence_mp4.append((S_dir, base_video_org_frame_dir))

        n_proc = len(sequence_mp4)
        n_proc = 64
        pool = mp.Pool(processes=n_proc)
        results = []

        for iseq, seq in enumerate(sequence_mp4):
            result = pool.apply_async(process_mp4_conductor, args=(seq[0], seq[1], iseq))
            results.append(result)
        print("start mp4 process.")
        pool.close()
        pool.join()
        sequence = [result.get() for result in results if result.get() is not None]

        ######################## Get Original Sequence ####################################

        sequence_original = []
        user_id_list = os.listdir(save_org_frame_dir)
        for U in tqdm(user_id_list, desc="User"):
            U_dir = os.path.join(save_org_frame_dir, U)
            US_list = os.listdir(U_dir)
            US_list = [x for x in US_list if (('_' in x) and ('.' not in x))]   # find folder
            os.makedirs(os.path.join(save_data_dir, U), exist_ok=True)
            for S in tqdm(US_list, desc="Scene"):
                S_dir = os.path.join(U_dir, S)
                base_save_dir = os.path.join(save_data_dir, U, S)
                os.makedirs(base_save_dir, exist_ok=True)
                rgb_list = os.listdir(S_dir)
                rgb_list.sort(key=extract_number)
                rgb_list = [os.path.join(S_dir, x) for x  in rgb_list]
                data = {
                    "id": f"egopat3d_{U}_{S}",
                    "U": U, "S": S,
                    "rgbs": rgb_list,
                }
                sequence_original.append(data)

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        #################### Preprare Sequential List for Parallel Preprocess ####################
        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "r") as f:
            sequence_original = json.load(f)
        
        sequence_preprocess = []
        for seq in tqdm(sequence_original):
            uid_path = os.path.join(save_data_dir, seq['U'], seq['S'])
            rgbs_list = seq['rgbs']
            n_images = len(rgbs_list)
            n_seq_data = 0

            seq_l = 0
            while seq_l + self.cfg.clip_frame <= n_images:
                seq_r = seq_l + self.cfg.clip_frame
                data_dict = {"id": seq['id']+"_"+str(n_seq_data), "save_dir": os.path.join(uid_path, str(n_seq_data)), "base_id": 0}
                rgbs_clip = []
                for i in range(seq_l, seq_r):
                    rgbs_clip.append(seq['rgbs'][i])
                data_dict['rgbs'] = rgbs_clip
                seq_l = seq_l + self.cfg.clip_frame
                sequence_preprocess.append(data_dict)
                n_seq_data = n_seq_data + 1
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

        train_id = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        val_id = ['14', '15']
        test_id = []
        sequence_val_0 = [x for x in sequence if x['id'].split('_')[1] in ['14']]
        n_val_0 = len(sequence_val_0)
        sequence_train = [x for x in sequence if x['id'].split('_')[1] in train_id] + sequence_val_0[:n_val_0//2]
        sequence_val = [x for x in sequence if x['id'].split('_')[1] in ['15']] + sequence_val_0[n_val_0//2:]
        sequence_test = [x for x in sequence if x['id'].split('_')[1] in test_id]

        with open(os.path.join(self.pre_save_root, "sequence_train.json"), "w") as f:
            json.dump(sequence_train, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_val.json"), "w") as f:
            json.dump(sequence_val, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_test.json"), "w") as f:
            json.dump(sequence_test, f, indent=4)

        print(f"len(train)={len(sequence_train)} ; len(val)={len(sequence_val)} ; len(test)={len(sequence_test)}")
        print(f"len_all={len(sequence)}")