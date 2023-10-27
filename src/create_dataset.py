import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np


class CodexHEDataset(Dataset):

    # Constructor method to initialize the dataset
    def __init__(self, root_dir: Path, mode: str = 'train', csv_file: str = "Annotations.csv", transform=None, augmentation=None):

        # Set the root directory
        self.root_dir = root_dir

        # Read annotations from a CSV file using pandas
        self.annotations = pd.read_csv(self.root_dir / csv_file)

        # Filter annotations based on the mode (train or test)
        self.annotations = self.annotations[self.annotations.iloc[:, 0].str.contains(mode)]

        # Set the transformation and augmentation functions
        self.transform = transform
        self.augmentation = augmentation

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.annotations)

    # Method to get a specific item from the dataset given an index
    def __getitem__(self, index: int):
        image_path, mask_path = self.annotations.iloc[index]

        # Load image and mask
        image = Image.open(self.root_dir / image_path).convert('RGB')
        mask = Image.open(self.root_dir / mask_path).convert('L')

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        if self.augmentation:
            augmented = self.augmentation(image=np.array(image), mask=np.array(mask))
            image, mask = augmented['image'], augmented['mask']

        # Convert to PyTorch tensors
        image = torch.tensor(np.array(image)).permute(0, 1, 2)
        mask = torch.tensor(np.array(mask)).permute(0, 1, 2)

        return image, mask
