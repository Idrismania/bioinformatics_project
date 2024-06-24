import os
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from dataset import UNetDataset
from hydra import initialize, compose
from transformations import combined_transform, to_tensor

with initialize(config_path="../conf", version_base='1.3'):
        cfg = compose(config_name="config.yaml")

def load_dataloaders():

    batch_size = cfg.params.batch_size

    # Set the path to your dataset
    root_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    img_dir = os.path.join(root_dir, "data", "epithelial", "he")
    mask_dir = os.path.join(root_dir, "data", "epithelial", "masks")

    dataset = UNetDataset(img_dir=img_dir, mask_dir=mask_dir, transform=None)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    torch.manual_seed(44)

    # Create datasets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


    train_dataset = Subset(UNetDataset(img_dir=img_dir, mask_dir=mask_dir, transform=combined_transform), train_dataset.indices)
    val_dataset = Subset(UNetDataset(img_dir=img_dir, mask_dir=mask_dir, transform=to_tensor), val_dataset.indices)
    test_dataset = Subset(UNetDataset(img_dir=img_dir, mask_dir=mask_dir, transform=to_tensor), test_dataset.indices)


    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
        pass
