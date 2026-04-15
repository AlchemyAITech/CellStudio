import os

import torch
from omegaconf import DictConfig

from cellstudio.backends.registry import BackendAdapterRegistry


class Trainer:
    """
    Unified Trainer interface for CellStudio.
    Automatically routes the task to configured backends (e.g., YoloAdapter, UnetAdapter).
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.backend
        
        # GPU Support Detection
        self.device = config.env.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Initializing with device: {self.device.upper()}")
        if self.device == "cuda":
            try:
                print(f"[Trainer] CUDA Device Name: {torch.cuda.get_device_name(0)}")
            except AssertionError:
                print("[Trainer] WARNING: Torch requested CUDA but compiled without it. Falling back to CPU.")
                self.device = "cpu"
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Build Trainer from loaded OmegaConf config."""
        return cls(config)
        
    def train(self):
        print(f"[{self.task_type.upper()}] Initiating '{self.backend_name}' backend training on {self.device}...")
        
        # Instantiate correct adapter from registry
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config, device=self.device)
        
        # System-Level Hook Injection
        # E.g. attach the RemoteProgressHook if we are running in a deployed environment
        from cellstudio.engine.hooks import RemoteProgressHook
        job_id = self.config.get("job_id", "local_run")
        progress_hook = RemoteProgressHook(job_id=job_id)
        adapter.register_hook(progress_hook)
        
        # Engine logic for finding data path expects dictconfig.data.data_dir or data.train_path
        if hasattr(self.config.data, "train_path"):
            data_path = self.config.data.train_path
        elif hasattr(self.config.data, "data_dir"):
            json_path = os.path.join(self.config.data.data_dir, "dataset.json") # Default standard name
            yaml_path = os.path.join(self.config.data.data_dir, "data.yaml")
            
            import glob
            existing_jsons = glob.glob(os.path.join(self.config.data.data_dir, "*standard.json"))
            
            if os.path.exists(json_path):
                data_path = json_path
            elif existing_jsons:
                data_path = existing_jsons[0] # Take the first found standard JSON
            elif os.path.exists(yaml_path):
                data_path = yaml_path
            else:
                data_path = self.config.data.data_dir
        else:
            raise KeyError("Config missing data.train_path or data.data_dir")
            
        print(f"[{self.task_type.upper()}] Resolved dataset path: {data_path}")
        result = adapter.train(data_path=data_path)
        
        print(f"\n[{self.task_type.upper()}] Training successfully completed!")
        print(f"Results saved at: {result.get('save_dir')}")
        return result

