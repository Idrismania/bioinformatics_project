"""
Given directories of training images and labels, a training sample index and number of augmentations,
this script plots a range of augmentations. This is useful for testing during augmentation implementations.
"""

import os
import sys
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from transformations import combined_transform
# Parameters
img_nr = 605
num_augmentations = 8

# Directories
img_dir = r"D:\Users\Horlings\ii_hh\bioinformatics_project\data\he"
mask_dir = r"D:\Users\Horlings\ii_hh\bioinformatics_project\data\masks"

# Load image and mask
img_name = os.listdir(img_dir)[img_nr]
mask_name = os.listdir(mask_dir)[img_nr]

img_path = os.path.join(img_dir, img_name)
mask_path = os.path.join(mask_dir, mask_name)

img = tifffile.imread(img_path)
mask = tifffile.imread(mask_path)[:, :, 6]

img = np.array(img, dtype=np.uint8)
mask = np.array(mask, dtype=np.uint8)

# Plotting function
def plot_augmentations(img, mask, num_augmentations):
    fig, axes = plt.subplots(num_augmentations, 2, figsize=(10, num_augmentations * 5))
    
    for i in range(num_augmentations):
        # Apply transformation
        augmented_img, augmented_mask = combined_transform(img, mask)
        
        # Convert to numpy for plotting
        augmented_img_np = augmented_img.permute(1, 2, 0).numpy()
        augmented_mask_np = augmented_mask.squeeze().numpy()
        
        # Plot image
        axes[i, 0].imshow(augmented_img_np)
        axes[i, 0].axis('off')
        
        # Plot mask
        axes[i, 1].imshow(augmented_mask_np, cmap='gray')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Plot the augmentations
plot_augmentations(img, mask, num_augmentations)
