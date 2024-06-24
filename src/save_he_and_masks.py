"""
Given data folder and output path, this script saves many augmented training images and mask labels side-by-side.
Useful for ensuring the model loads proper images and their appropriate, correctly transformed masks. 
"""

import numpy as np
import tifffile 
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

path = Path(r"E:\Users\Horlings\ii_hh\bioinformatics_project\data")
out_path = Path(r"E:\\Users\\Horlings\\ii_hh\\bioinformatics_project\\input_label_pairs")

# Geometric transformations (applied to both image and mask)
geometric_transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.75),
    A.Rotate(p=1)
], additional_targets={'mask': 'mask'})

# Color transformations (applied only to the image)
color_transforms = A.Compose([
    A.GaussianBlur(blur_limit=(3, 9), p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.15), contrast_limit=0.15, p=0.9),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0, p=1.0)
])

# Function to apply geometric transforms to both image and mask, and color transforms to the image
def transform(image, mask):
    # Apply geometric transforms

    augmented = geometric_transforms(image=image, mask=mask)

    image = augmented['image']

    mask = augmented['mask']

    # Apply color transforms only to the image
    image = color_transforms(image=image)['image']

    return image, mask

# Combined transformation function
def combined_transform(image, mask):
    image, mask = transform(image, mask)
    return image, mask


for i in range(len(os.listdir(r"E:\Users\Horlings\ii_hh\bioinformatics_project\data\he"))):

    img_path = os.path.join(path, "he", f"img{i}.tif")
    mask_path = os.path.join(path, "masks", f"img{i}_mask.tif")
            
    image = tifffile.imread(img_path)
    mask = tifffile.imread(mask_path)[:, :, 0] # THIS INDEX DECIDES WHICH CHANNEL TO LOAD, VERY IMPORTANT

    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask*255, dtype=np.uint8)

    image, mask = combined_transform(image, mask)

    mask_rgb = np.stack((mask,)*3, axis=-1)

    fig = np.concatenate((image, mask_rgb), axis=1)
    im = Image.fromarray(fig)
    im.save(Path(out_path, f"image_{i}.png"))


    