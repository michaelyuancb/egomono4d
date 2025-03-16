from dataclasses import dataclass
import torch
import random
import pdb
import os
import re
import json
import cv2
import numpy as np
from tqdm import tqdm
from einops import einsum, rearrange
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tf
from ..misc.data_util import pil_resize_to_center_crop, compute_patch_cropped_shape


class DatasetCustom(Dataset):

    def __init__(self, cfg, args, max_frame=360) -> None:
        
        self.resize_shape = cfg.preprocess.resize_shape
        self.patch_shape = compute_patch_cropped_shape(self.resize_shape, cfg.preprocess.patch_size)

        step_overlap = args.step_overlap
        frames_dir = args.frames_dir
        cache_folder = frames_dir.replace('/', '_')
        cache_dir = os.path.join(args.cache_dir, str(self.resize_shape)+"_"+str(self.patch_shape)+cache_folder)
        cache_dir = cache_dir.replace(',', '_').replace(' ', '').replace('(','').replace(')','')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            self.sequence = self._preprocess(frames_dir, cache_dir, cfg, max_frame=max_frame)
        else:
            with open(os.path.join(cache_dir, "frame_sequence.json"), "r") as f:
                self.sequence = json.load(f)
        
        n_images = len(self.sequence)
        self.num_frames = cfg.preprocess.num_frames
        seq_l_last = (n_images-self.num_frames) // (self.num_frames-step_overlap)
        seq_l_list = [i*(self.num_frames-step_overlap) for i in range(seq_l_last+1)]
        if seq_l_list[-1] + self.num_frames < n_images:
            seq_l_list.append(n_images - self.num_frames)
        self.seq_l_list = seq_l_list

        seq_r_last = self.seq_l_list[-1] + self.num_frames
        if seq_r_last != len(self.sequence):
            print(f"For {len(self.sequence)} frames and {step_overlap} step-interval, seq_r={seq_r_last}")
        print(f"Finish Generating Sequence Batch, Number-of-Sequence: {len(self.seq_l_list)}")


    def _preprocess(self, frames_dir, cache_dir, cfg, max_frame=400):     # resize / crop frames to self.patch_shape

        def extract_number(filename):
            filename = filename.split('/')[-1]
            match = re.search(r'\d+', filename)  
            return int(match.group()) if match else 0  

        rgbs_list = []
        if frames_dir.endswith(".mp4"):
            cap = cv2.VideoCapture(frames_dir)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for idx in tqdm(range(frame_count), desc="Read Frames from Video"):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbs_list.append(Image.fromarray(rgb_frame))
                if len(rgbs_list) == max_frame:
                    break
            cap.release()
        else:
            frames_list = os.listdir(frames_dir)
            frames_list.sort(key=extract_number)
            assert len(frames_list) <= max_frame
            for idx, frame_fp in tqdm(enumerate(frames_list), desc="Read Frames"):
                rgb = Image.open(os.path.join(frames_dir, frame_fp))
                rgbs_list.append(rgb)
        
        frame_sequence = []
        for idx, rgb in tqdm(enumerate(rgbs_list), desc="Resize Frames"):
            rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, self.resize_shape, self.patch_shape)
            rgb_fp = os.path.join(cache_dir, "frame"+str(idx)+".png")
            rgb.save(rgb_fp)
            frame_sequence.append(rgb_fp)
        with open(os.path.join(cache_dir, "frame_sequence.json"), "w") as f:
            json.dump(frame_sequence, f, indent=4)
        return frame_sequence


    def __getitem__(self, index: int):
        seq_l = self.seq_l_list[index]
        seq_r = seq_l + self.num_frames
        rgb_fp_list = [self.sequence[i] for i in range(seq_l, seq_r)]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgb_fp_list])
        data = {"videos": videos, "start_indice": seq_l}
        return data


    def __len__(self) -> int:
        return len(self.seq_l_list)