from model_architecture import UNet
from load_data import load_dataloaders
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


    # Main function to load model, make predictions, and plot results
def main(model_path, test_loader):
    # Load the model
    print("loading model...")
    model = load_model(model_path)
    
    print("retrieving data...")
    # Get a batch of test data
    inputs, labels = next(iter(test_loader))
    
    # Move data to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("moving data to cpu...")
    model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)
    
    print("performing model inference...")
    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)


    print("preparing data for plotting...")

    # Convert predictions and labels to CPU numpy arrays for plotting
    inputs = inputs.cpu().numpy()[2]
    predictions = outputs.cpu().numpy()[2]
    labels = labels.cpu().numpy()[2]

    input_img = np.transpose(inputs, (1, 2, 0))
    prediction = np.transpose(predictions, (1, 2, 0))
    label = np.transpose(labels, (1, 2, 0))
    
    print("plotting data...")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # Plot input image
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot prediction
    axes[1].imshow(prediction)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Plot label
    axes[2].imshow(label)
    axes[2].set_title('Label')
    axes[2].axis('off')

    axes[3].imshow(np.where(prediction >= 0.25, 1, 0))
    axes[3].set_title('Processed inference')
    axes[3].axis('off')
    
    plt.show()

if __name__ == "__main__":
    # Replace these with your paths and DataLoader


    model_path = r"E:\Users\Horlings\ii_hh\bioinformatics_project\output_model\model.pth"

    
    # Define your test dataset and DataLoader
    # Example:
    # test_dataset = UNetDataset(img_dir='path/to/test/images', mask_dir='path/to/test/masks', transform=your_transform)
    # test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=4)
    train_dataloader, val_dataloader, test_dataloader = load_dataloaders()
    # Call the main function with the model path and test loader
    main(model_path, train_dataloader)