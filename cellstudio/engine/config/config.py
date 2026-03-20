import os
from pathlib import Path
from typing import Union, List, Dict, Any
from omegaconf import OmegaConf, DictConfig

class Config:
    """
    The Zenith Config Engine.
    Provides deep merging, variable interpolation, and MM-style `_base_` inheritance.
    """
    @staticmethod
    def _merge_base(cfg_path: str) -> DictConfig:
        cfg_path = Path(cfg_path).absolute()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
            
        cfg = OmegaConf.load(cfg_path)
        
        # Check for _base_ inheritance
        if '_base_' in cfg:
            bases = cfg['_base_']
            if isinstance(bases, str):
                bases = [bases]
                
            merged_base = OmegaConf.create()
            for base_file in bases:
                base_path = cfg_path.parent / base_file
                # Recursively parse and merge base files
                base_cfg = Config._merge_base(str(base_path))
                merged_base = OmegaConf.merge(merged_base, base_cfg)
                
            # Remove _base_ key, merge current cfg on top of the imported bases
            del cfg['_base_']
            cfg = OmegaConf.merge(merged_base, cfg)
            
        return cfg

    @staticmethod
    def fromfile(cfg_path: str) -> DictConfig:
        """
        Loads a config file, meticulously resolving _base_ inheritance 
        and OmegaConf interpolations.
        """
        cfg = Config._merge_base(cfg_path)
        # Eagerly resolve interpolations (e.g. ${task.type})
        OmegaConf.resolve(cfg)
        return cfg
