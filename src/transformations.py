import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

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

# Function to convert images and masks to tensors
def to_tensor(image, mask):

    image = np.array(image, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.float32)

    image = ToTensorV2()(image=image)['image']
    mask = ToTensorV2()(image=mask)['image']
    return image, mask

# Combined transformation function
def combined_transform(image, mask):
    image, mask = transform(image, mask)
    image, mask = to_tensor(image, mask)
    return image, mask
