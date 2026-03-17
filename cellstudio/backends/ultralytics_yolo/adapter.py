import os
from omegaconf import DictConfig
from pathstudio.backends.base.adapter import BaseBackendAdapter

class YoloAdapter(BaseBackendAdapter):
    """
    Adapter for Ultralytics YOLOv8/v11.
    Wraps the YOLO class to conform strictly to the PathStudio BaseBackendAdapter protocol.
    """
    
    def __init__(self, config: DictConfig):
        # We delay import specifically so that if ultralytics is not installed, 
        # PathStudio core won't crash unless YOLO is strictly requested.
        try:
            from ultralytics import YOLO
            self._YOLO_CLASS = YOLO
        except ImportError:
            raise ImportError("Ultralytics library is required for YoloAdapter. Please `pip install ultralytics`")
            
        super().__init__(config)

    def _build_model(self):
        """Initializes YOLO model from pretrained weights or architecture yaml."""
        model_arch = self.config.model.architecture
        weights_path = self.config.model.pretrained_weights
        
        # Load custom weights if provided, otherwise load architecture structure
        if weights_path and os.path.exists(weights_path):
            return self._YOLO_CLASS(weights_path)
        else:
            # e.g., 'yolov8n.pt' or 'yolov8n.yaml'
            target = f"{model_arch}.pt" if self.config.model.pretrained else f"{model_arch}.yaml"
            return self._YOLO_CLASS(target)

    def train(self, data_path: str, **kwargs) -> dict:
        """
        Executes YOLO training process.
        Maps PathStudio's unified train configs to YOLO arguments.
        Allows passing JSON data schema instead of tight-coupled YAML.
        """
        # If the input is our PathStudio JSON, format it to YOLO first
        if data_path.endswith('.json'):
            from pathstudio.backends.ultralytics_yolo.formatter import YoloDataFormatter
            formatter = YoloDataFormatter(output_dir=self.config.training.save_dir)
            data_yaml_path = formatter.format_from_json(data_path, task_type=self.config.task)
        else:
            data_yaml_path = data_path
            
        train_args = {
            "data": data_yaml_path,
            "epochs": self.config.training.epochs,
            "batch": self.config.data.batch_size,
            "lr0": self.config.training.learning_rate,
            "device": self.config.env.device,
            "project": self.config.training.save_dir,
            "name": f"{self.config.task}_{self.config.model.architecture}",
            "exist_ok": True,
        }
        
        # Override with any custom kwargs directly passed via Engine layer
        train_args.update(kwargs)
        
        # Bridge YOLO's internal callbacks to PathStudio's HookManager
        def on_train_epoch_end(trainer):
            metrics = {k: v for k, v in trainer.metrics.items()}
            self.hook_manager.trigger("on_epoch_end", self, epoch=trainer.epoch, metrics=metrics)
            
        def on_train_start(trainer):
            self.hook_manager.trigger("on_train_begin", self)

        def on_train_end(trainer):
            self.hook_manager.trigger("on_train_end", self)

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.model.add_callback("on_train_start", on_train_start)
        self.model.add_callback("on_train_end", on_train_end)
        
        results = self.model.train(**train_args)
        
        # Standardize return format based on YOLO's results object
        return {
            "status": "success",
            "save_dir": os.path.join(train_args["project"], train_args["name"]),
            "metrics": results.results_dict if hasattr(results, "results_dict") else {}
        }
        
    def evaluate(self, data_path: str, **kwargs) -> dict:
        """Executes YOLO evaluation."""
        if data_path.endswith('.json'):
            from pathstudio.backends.ultralytics_yolo.formatter import YoloDataFormatter
            formatter = YoloDataFormatter(output_dir=self.config.training.save_dir)
            data_yaml_path = formatter.format_from_json(data_path, task_type=self.config.task)
        else:
            data_yaml_path = data_path
            
        eval_args = {
            "data": data_yaml_path,
            "device": self.config.env.device
        }
        eval_args.update(kwargs)
        
        metrics = self.model.val(**eval_args)
        return {
            "map50-95": metrics.box.map if hasattr(metrics, "box") else 0.0,
            "map50": metrics.box.map50 if hasattr(metrics, "box") else 0.0
        }

    def predict(self, source: str, **kwargs):
        """Executes YOLO inference."""
        predict_args = {
            "source": source,
            "device": self.config.env.device,
            "save": True, # Save visualizations by default for MVP
            "project": self.config.training.save_dir,
            "name": "predict_results",
            "exist_ok": True
        }
        predict_args.update(kwargs)
        
        results = self.model.predict(**predict_args)
        # Translate YOLO specific result objects to standard format if needed
        # For now, return YOLO Results generator
        return results

    def export(self, export_format: str, save_path: str = None, **kwargs) -> str:
        """Exports YOLO model to requested format."""
        
        # format mapping if ultralytics uses different names
        fmt_map = {"onnx": "onnx", "tensorrt": "engine", "torchscript": "torchscript"}
        target_fmt = fmt_map.get(export_format.lower(), export_format)
        
        export_args = {
            "format": target_fmt,
            "device": "cpu" if target_fmt in ["onnx", "torchscript"] else self.config.env.device,
            "opset": 17 if target_fmt == "onnx" else None
        }
        export_args.update(kwargs)
        
        saved_file = self.model.export(**export_args)
        return str(saved_file)
