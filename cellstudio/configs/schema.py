from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass
class EnvironmentConfig:
    device: str = "cuda"
    seed: int = 42

@dataclass
class DataConfig:
    dataset_name: str = "custom"
    data_dir: str = "./data"
    batch_size: int = 16
    num_workers: int = 4

@dataclass
class ModelConfig:
    backend: str = "yolo"           # e.g. yolo, unet, timm
    architecture: str = "yolov8n"   # e.g. yolov8n, resnet50
    pretrained: bool = True
    pretrained_weights: str = ""
    in_channels: int = 3
    export_format: str = "onnx"

@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "AdamW"
    save_dir: str = "./runs"

@dataclass
class PathStudioConfig:
    task: str = "detection"         # classification, detection, segmentation
    env: EnvironmentConfig = EnvironmentConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainConfig = TrainConfig()

def get_default_config() -> DictConfig:
    """Returns the structured default configuration schema."""
    return OmegaConf.structured(PathStudioConfig)

def load_config(cfg_path: str) -> DictConfig:
    """Loads a yaml config and merges it with the strict schema."""
    schema = get_default_config()
    user_cfg = OmegaConf.load(cfg_path)
    return OmegaConf.merge(schema, user_cfg)
