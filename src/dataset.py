import os
import pandas as pd
import torch
import tifffile
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from transformations import to_tensor


class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".tif", "_mask.tif"))  # assumes mask file names are similar
        
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)[:, :, :] # THIS INDEX DECIDES WHICH CHANNEL TO LOAD, VERY IMPORTANT


        mask = mask[:, :, 3] + mask[:, :, 4] + mask[:, :, 5]
        mask = np.where(mask > 0, 1, 0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

if __name__ == "__main__":
    pass
