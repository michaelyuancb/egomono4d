import torch
import random
import pdb
from typing import List
from .types import Stage
from torch.utils.data import Dataset


class DatasetMerged(Dataset):

    def __init__(self, 
                 datasets: List[Dataset],
                 stage: Stage,
                 global_rank: int,
                 world_size: int,
                 data_ratio: float=1.0
                ) -> None:
        self.datasets = datasets
        self.stage = stage
        self.global_rank = global_rank
        self.world_size = world_size
        index_list = []
        
        for ids, dataset in enumerate(self.datasets):
            index_list = index_list + [(ids, i) for i in range(int(len(dataset)*data_ratio))]
        
        random.seed(0)
        random.shuffle(index_list)
        self.index_list = index_list

        print(f"################### [Stage {stage}: Num Data = {len(self.index_list)}] ###################")


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        dataset_id, data_id = self.index_list[index]
        # print(f"[Data Go] global_rank={self.global_rank} | dataloader_index={index} | data_index={(dataset_id, data_id)}")
        return self.datasets[dataset_id][data_id]
