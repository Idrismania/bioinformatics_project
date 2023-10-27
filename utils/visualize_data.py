import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


SAMPLE_INDEX = 1

# Retrieve list of image paths
images = os.listdir('../data/test/image')
masks = os.listdir('../data/test/mask')

# Construct image path from indexed samples
img_path = Path('../data/test/image/' + images[SAMPLE_INDEX])
mask_path = Path('../data/test/mask/' + masks[SAMPLE_INDEX])

# PIL open images
image = Image.open(img_path)
mask = Image.open(mask_path)

# Create 1x2 subplot with image on the left and mask on the right
fig, axes = plt.subplots(1, 2, figsize=(4, 2))

axes[0].imshow(image)
axes[1].imshow(mask)

plt.tight_layout()
plt.show()
