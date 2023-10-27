from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

resize_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


mask = Image.open('../data/train/mask/cxrmask_0.jpeg').convert('L')
mask.show()
mask = resize_transform(mask)

mask = np.array(mask)  # Convert to NumPy array



mask = torch.tensor(mask)
mask = mask.permute(1, 2, 0)

print(mask)

plt.imshow(mask, cmap='grey')
plt.show()
