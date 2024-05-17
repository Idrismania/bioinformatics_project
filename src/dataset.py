import os
import pandas as pd
import torch
import tifffile
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".tif", "_mask.tif"))  # assumes mask file names are similar
        
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)[:, :, 0] # THIS INDEX DECIDES WHICH CHANNEL TO LOAD, VERY IMPORTANT

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
