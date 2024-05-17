import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

img_nr = 605

img_dir = r"D:\Users\Horlings\ii_hh\bioinformatics_project\data\he"
mask_dir = r"D:\Users\Horlings\ii_hh\bioinformatics_project\data\masks"

image = os.listdir(img_dir)[img_nr]
mask = os.listdir(mask_dir)[img_nr]

img = tifffile.imread(os.path.join(img_dir, image))
mask = tifffile.imread(os.path.join(mask_dir, mask))

print(img.shape)
print(mask.shape)

plt.imshow(mask[:, :, 0:3])
plt.show()

print(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))