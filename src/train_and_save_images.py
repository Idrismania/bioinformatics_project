import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import numpy as np
from pathlib import Path
from hydra.core.config_store import ConfigStore
from config import UnetConfig
from tqdm import tqdm
import cv2
# Own scripts
from model_architecture import AttentionUNet
from load_data import load_dataloaders

cs = ConfigStore.instance()
cs.store('UnetConfig', node=UnetConfig)



# Custom Jaccard loss function
def jaccard_index_loss(predictions, targets):
    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions) + torch.sum(targets) - intersection
    smooth = 1e-6  # smoothing factor to avoid division by zero
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou


@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def main(cfg: UnetConfig):

    train_dataloader, val_dataloader, test_dataloader = load_dataloaders()
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently training on {device}:")
    
    inputs, labels = next(iter(val_dataloader))
    inputs, labels = inputs.to(device), labels.to(device)


    # Hyperparameters
    learning_rate = cfg.params.learning_rate
    num_epochs = cfg.params.epoch_count

    # Instantiate the SimpleCNN model
    model = AttentionUNet()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.to(device)
    iterator = 1
    # Training loop with tqdm
    for epoch in range(num_epochs):

        

        running_loss = 0.0

        tqdm_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        # tqdm for the dataloader
        for images, labels in tqdm_dataloader:

            torch.backends.cudnn.benchmark = True
            model.train()  # Set the model to training mode

            # Move data to the device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Save output in case you want to inspect
            torch.save(outputs, Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')), "output_model", "output.pt"))

            # Compute the loss
            loss = jaccard_index_loss(outputs, labels)

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Update tqdm progress bar with the current loss
            tqdm_dataloader.set_postfix(loss=loss.item())
            
            model.eval()
            outputs = model(inputs)
            predictions = outputs.cpu().detach().numpy()[2]

            prediction = np.transpose(predictions, (1, 2, 0))

            im = (prediction * 255).astype(np.uint8)
            cv2.imwrite(f"E:/Users/Horlings/ii_hh/bioinformatics_project/output_model/images/jaccard/iteration_{iterator}.png", im)
            iterator += 1

if __name__ == "__main__":
    main()
