import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from pathlib import Path
from hydra.core.config_store import ConfigStore
from config import UnetConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# Own scripts
from model_architecture import UNet
from load_data import load_dataloaders

cs = ConfigStore.instance()
cs.store('UnetConfig', node=UnetConfig)


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         # Flatten the tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         # Calculate intersection and union
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
#         return 1 - dice

# # Custom Jaccard loss function
# def jaccard_index_loss(predictions, targets):
#     intersection = torch.sum(predictions * targets)
#     union = torch.sum(predictions) + torch.sum(targets) - intersection
#     smooth = 1e-6  # smoothing factor to avoid division by zero
#     iou = (intersection + smooth) / (union + smooth)
#     return 1 - iou

@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def main(cfg: UnetConfig):
    scaler = torch.cuda.amp.GradScaler()
    train_dataloader, _, _ = load_dataloaders()

    # Hyperparameters
    learning_rate = cfg.params.learning_rate
    num_epochs = cfg.params.epoch_count

    # Instantiate the SimpleCNN model
    model = UNet().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently training on {device}:")
    model.to(device)

    # Training loop with tqdm
    for epoch in range(num_epochs):

        torch.backends.cudnn.benchmark = True
        model.train()  # Set the model to training mode

        running_loss = 0.0

        tqdm_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        # tqdm for the dataloader
        for images, labels in tqdm_dataloader:

            with torch.cuda.amp.autocast():
                # Move data to the device (GPU or CPU)
                images, labels = images.to(device), labels.to(device)


                with torch.autocast(device_type='cuda', dtype=torch.float16):

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    writer.add_scalar("Loss/train", loss, epoch)

                # Zero the gradients
                optimizer.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)

                scaler.update()

                # Update the running loss
                running_loss += loss.item()

                # Update tqdm progress bar with the current loss
                tqdm_dataloader.set_postfix(loss=loss.item())

                del loss
                del outputs
            
            # Save the trained model
        torch.save(model.state_dict(), Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')), "output_model", f"model.pth"))
        writer.flush()

if __name__ == "__main__":
    writer = SummaryWriter()
    main()
