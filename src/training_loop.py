import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from pathlib import Path
from hydra.core.config_store import ConfigStore
from config import UnetConfig
from tqdm import tqdm

# Own scripts
from model_architecture import R2AttU_Net
from load_data import train_dataloader

cs = ConfigStore.instance()
cs.store('UnetConfig', node=UnetConfig)

@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def main(cfg: UnetConfig):

    # Hyperparameters
    learning_rate = cfg.params.learning_rate
    num_epochs = cfg.params.epoch_count

    # Instantiate the SimpleCNN model
    model = R2AttU_Net(output_ch=1)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently training on {device}:")
    model.to(device)

    # Training loop with tqdm
    for epoch in range(num_epochs):

        model.train()  # Set the model to training mode

        running_loss = 0.0

        tqdm_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        # tqdm for the dataloader
        for images, labels in tqdm_dataloader:

            # import matplotlib.pyplot as plt

            # # Assuming images and labels are tensors from the first batch
            # first_image = images[0].permute(1, 2, 0)
            # first_label = labels[0]
            # # Plot the first image and label side by side
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # # Plot the first image
            # axes[0].imshow(first_image)
            # axes[0].set_title('Image')
            
            # # Plot the first label
            # axes[1].imshow(first_label[0])
            # axes[1].set_title('Label')
            
            # plt.show()

            # Move data to the device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Save output in case you want to inspect
            torch.save(outputs, Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')), "output_model", "output.pt"))

            # Compute the loss
            loss = criterion(outputs, labels)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Update tqdm progress bar with the current loss
            tqdm_dataloader.set_postfix(loss=loss.item())
            
            # Save the trained model
            torch.save(model.state_dict(), Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')), "output_model", "model.pth"))

if __name__ == "__main__":
    main()
