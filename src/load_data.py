from torch.utils.data import DataLoader
from pathlib import Path
from create_dataset import CodexHEDataset
# from augmentations import train_augmentation
from transformations import resize_transform

# Set the path to your dataset
root_dir = Path('../')

train_dataset = CodexHEDataset(root_dir, csv_file='data.csv', mode='train', transform=resize_transform, augmentation=None)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = CodexHEDataset(root_dir, csv_file='data.csv', mode='test', transform=resize_transform, augmentation=None)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
