import os
from hydra import initialize, compose

# Import hydra configuration
with initialize(config_path="../conf", version_base='1.3'):
    cfg = compose(config_name="config.yaml")

# Add necessary dll directories
openslide_path = cfg.paths.openslide
vipshome = cfg.paths.vips

os.add_dll_directory(openslide_path)
os.add_dll_directory(vipshome)

# import libraries
import pyvips
import dlup

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup import SlideImage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from dlup.background_deprecated import get_mask
from dlup.data.dataset import Grid
from pathlib import Path
from matplotlib import pyplot as plt

INPUT_FILE_PATH = Path('../wsi_data/CMU-1.svs')
slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)

TILE_SIZE = (cfg.tiling_params.tile_size_x, cfg.tiling_params.tile_size_y)
TARGET_MPP = cfg.tiling_params.microns_per_pixel

# Generate the mask
mask = get_mask(slide_image)

# Input, MPP, Tile_Size, Tile_Overlap
"""Idea from Bart to make dataset of all images: data = [dataset1, dataset2, dataset3] etc.
Also, MPP is not unique per image. It is the MPP you want to scale all your images to for consistency.
0.5 seems standard."""
dataset = TiledROIsSlideImageDataset.from_standard_tiling(INPUT_FILE_PATH, 1, TILE_SIZE, (0, 0), mask=mask)


for i, d in enumerate(dataset):
    plt.imshow(d['image'])
    plt.title(f"{i}/{len(dataset)}")
    plt.show()



# grid1 = Grid.from_tiling(
#     (0, 0), # offset x, y
#     size=(250, 450), # height, width
#     tile_size=TILE_SIZE,
#     tile_overlap=(0, 0)
# )
#
# dataset = TiledROIsSlideImageDataset(INPUT_FILE_PATH, [(grid1, TILE_SIZE, TARGET_MPP)], mask=mask)
