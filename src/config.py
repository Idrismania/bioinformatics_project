from dataclasses import dataclass
from typing import Union
@dataclass
class Paths:
    data: str
    openslide: str
    vips: str

@dataclass
class Params:
    epoch_count: int
    learning_rate: float
    batch_size: int
    channel: int

@dataclass
class Tiling_Params:
    tile_size_x: int
    tile_size_y: int
    microns_per_pixel: Union[float, int]

@dataclass
class UnetConfig:
    paths: Paths
    params: Params
    tiling_params: Tiling_Params
