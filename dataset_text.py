from utils.lmdb_utils import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


class TextDataset_json(Dataset):
    def __init__(self, prompt_path):
        with open(prompt_path, "r") as f:
            self.metadata = json.load(f)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        batch = self.metadata[idx]
        batch = {
            "prompts": batch,
            "idx": idx,
        }
        return batch



    


if __name__ == "__main__":
    dataset = TextDataset_json(
        prompt_path="/hpc2hdd/home/htian395/Wenxue/Self-Forcing-Long/data/ultralong_32_extracted.json"
    )
    for data in dataset:
        print(data)
        break