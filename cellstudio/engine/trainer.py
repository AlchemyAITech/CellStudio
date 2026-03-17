import os
from omegaconf import DictConfig
from pathstudio.backends.registry import BackendAdapterRegistry

class Trainer:
    """
    Unified Trainer interface for PathStudio.
    Automatically routes the task to configured backends (e.g., YoloAdapter, UnetAdapter).
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.model.backend
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Build Trainer from loaded OmegaConf config."""
        return cls(config)
        
    def train(self):
        print(f"[{self.task_type.upper()}] Initiating '{self.backend_name}' backend training...")
        
        # Instantiate correct adapter from registry
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config)
        
        # System-Level Hook Injection
        # E.g. attach the RemoteProgressHook if we are running in a deployed environment
        from pathstudio.engine.hooks import RemoteProgressHook
        job_id = self.config.get("job_id", "local_run")
        progress_hook = RemoteProgressHook(job_id=job_id)
        adapter.register_hook(progress_hook)
        
        # Data logic: Prefer PathStudio unified JSON over fallback data.yaml
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
            # Flexible: let the adapter try to resolve the data_dir itself if specific filenames missing
            data_path = self.config.data.data_dir
            
        print(f"[{self.task_type.upper()}] Resolved dataset path: {data_path}")
        result = adapter.train(data_path=data_path)
        
        print("\n[{self.task_type.upper()}] Training successfully completed!")
        print(f"Results saved at: {result.get('save_dir')}")
        return result

