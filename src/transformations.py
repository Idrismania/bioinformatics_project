import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import myTransforms as B

# Geometric transformations (applied to both image and mask)
geometric_transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1),
    A.Rotate(limit=(-45, 45), p=1),
    #A.ElasticTransform(alpha=80, sigma=9, alpha_affine=20, p=1)
], additional_targets={'mask': 'mask'})

# Color transformations (applied only to the image)
color_transforms = A.Compose([
    A.GaussianBlur(blur_limit=(3, 9), p=0.2)
])
    

# Function to apply geometric transforms to both image and mask, and color transforms to the image
def transform(image, mask):
    # Apply geometric transforms

    augmented = geometric_transforms(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    # Apply color transforms only to the image
    image = color_transforms(image=image)['image']

    # HE augmentation from Tellez, D., Balkenhol, M., Otte-Höller, I., van de Loo, R., Vogels, R., Bult, P., ... & Litjens, G. (2018).
    # Whole-slide mitosis detection in H&E breast histology using PHH3 as a reference to train distilled stain-invariant convolutional 
    HE_augment = B.HEDJitter(theta=0.03)
    image = HE_augment(image)

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


if __name__ == "__main__":
    pass
