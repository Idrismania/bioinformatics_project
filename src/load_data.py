from torch.utils.data import DataLoader
from pathlib import Path
from create_dataset import CodexHEDataset
from hydra import initialize, compose
from transformations import resize_transform

with initialize(config_path="../conf", version_base='1.3'):
    cfg = compose(config_name="config.yaml")

batch_size = cfg.params.batch_size

# Set the path to your dataset
root_dir = Path('../')

train_dataset = CodexHEDataset(root_dir, csv_file='data.csv', mode='train', transform=resize_transform, augmentation=None)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CodexHEDataset(root_dir, csv_file='data.csv', mode='test', transform=resize_transform, augmentation=None)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
