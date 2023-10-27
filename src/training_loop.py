import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_architecture import UNet
from load_data import train_dataloader

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Instantiate the SimpleCNN model
model = UNet(1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0

    tqdm_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

    # tqdm for the dataloader
    for images, labels in tqdm_dataloader:

        # Move data to the device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

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
torch.save(model.state_dict(), f'../output_model/CNN_basic_{num_epochs}_epochs.pth')
