from pathlib import Path
import numpy as np
from pylibCZIrw import czi as pyczi
import matplotlib.pyplot as plt
from PIL import Image

# Input czi
file = Path(r'../wsi_unconverted/Codex_Tissue.czi')

plane_1 = {'C': 0, 'Z': 0, 'T': 0}


with pyczi.open_czi(str(file)) as czidoc:

    bounding_box = czidoc.total_bounding_box
    my_roi = (bounding_box['X'][0],  # Offset X
              bounding_box['Y'][0],  # Offset, Y
              int(98117/4),                               # Size X
              int(55187/4))                               # Size Y

    frame_1 = czidoc.read(roi=my_roi, plane=plane_1)
    print(np.max(frame_1))

    #print(f"Bounding box: {bounding_box}")

# Convert to 8-bit
frame_1 = ((frame_1 / np.max(frame_1)) * 255).astype(np.uint8)

# Turn BGR to RGB
frame_1 = frame_1[:, :, ::-1]

img = Image.fromarray(frame_1)

img.save('../wsi_data/saved.tif')
