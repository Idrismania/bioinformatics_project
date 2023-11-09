from dataclasses import dataclass

@dataclass
class Paths:
    data: str

@dataclass
class Params:
    epoch_count: int
    learning_rate: float
    batch_size: int

@dataclass
class UnetConfig:
    paths: Paths
    params: Params
