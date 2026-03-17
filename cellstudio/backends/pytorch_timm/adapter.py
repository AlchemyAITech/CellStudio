import os
import tempfile
from omegaconf import DictConfig
from typing import Dict, Any

from pathstudio.backends.base.adapter import BaseBackendAdapter
from pathstudio.datasets.schema import CellDatasetConfig

class TimmAdapter(BaseBackendAdapter):
    """
    Adapter for PyTorch Image Models (timm).
    Provides classification training, evaluation, and export capabilities 
    conforming strictly to the PathStudio BaseBackendAdapter protocol.
    """
    def __init__(self, config: DictConfig):
        try:
            import timm
            import torch
            from torch import nn
            self.timm = timm
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("PyTorch and timm libraries are required for TimmAdapter. Please `pip install torch torchvision timm`")
            
        self.device = self.torch.device(config.env.device if self.torch.cuda.is_available() else "cpu")
        super().__init__(config)

    def _build_model(self):
        """Initializes TIMM model based on architecture defined in config."""
        model_arch = self.config.model.architecture
        pretrained = self.config.model.pretrained
        weights_path = self.config.model.pretrained_weights
        
        # Determine number of classes
        num_classes = 1000 
        state_dict_payload = None
        
        if weights_path and os.path.exists(weights_path):
            state_dict_payload = self.torch.load(weights_path, map_location=self.device)
            if 'classes' in state_dict_payload:
                num_classes = len(state_dict_payload['classes'])
                
        model = self.timm.create_model(model_arch, pretrained=pretrained, num_classes=num_classes)
        
        if state_dict_payload:
            # Basic state dict loading
            if 'model_state_dict' in state_dict_payload:
                model.load_state_dict(state_dict_payload['model_state_dict'])
            else:
                model.load_state_dict(state_dict_payload)
                
        return model.to(self.device)

    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Executes native PyTorch/Timm training loop driven by PathStudio configs.
        Expects a PathStudio JSON schema dataset.
        """
        if not data_path.endswith('.json'):
            raise ValueError(f"TimmAdapter expects a standard PathStudio .json schema, got {data_path}")
            
        print(f"[TimmAdapter] Loading dataset from {data_path}")
        dataset_config = CellDatasetConfig.load(data_path)
        classes = dataset_config.classes
        
        # Reset model head based on actual classes
        if len(classes) != self.model.num_classes:
            print(f"[TimmAdapter] Adjusting model head for {len(classes)} classes.")
            self.model.reset_classifier(len(classes))
            self.model = self.model.to(self.device)
            
        # Implementation of full PyTorch training loop
        print(f"[TimmAdapter] Preparing datasets and dataloaders...")
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image
        
        class CellJsonDataset(Dataset):
            def __init__(self, items, class_list, transform=None):
                self.items = items
                self.class_list = class_list
                self.transform = transform
                
            def __len__(self):
                return len(self.items)
                
            def __getitem__(self, idx):
                item = self.items[idx]
                img = Image.open(item.image_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                # Primary label assuming single label classification for Phase 3
                label_idx = self.class_list.index(item.cls_labels[0])
                return img, label_idx
                
        # Basic ImageNet transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = CellJsonDataset(dataset_config.items, classes, transform=transform)
        # Handle mac environment multiprocessing edge cases by defaulting to 0 for MVP
        workers = self.config.data.get("num_workers", 0) 
        train_loader = DataLoader(train_dataset, batch_size=self.config.data.batch_size, shuffle=True, num_workers=0)
        
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        criterion = self.nn.CrossEntropyLoss()
        
        print(f"[TimmAdapter] Starting training for {self.config.training.epochs} epochs")
        
        self.hook_manager.trigger("on_train_begin", self)
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Trigger step hook
                self.hook_manager.trigger("on_train_step_end", self, step=step, loss=loss.item())
                
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            
            print(f"[Epoch {epoch}/{self.config.training.epochs}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
            
            # Trigger epoch hook
            metrics = {"loss": epoch_loss, "accuracy": epoch_acc}
            self.hook_manager.trigger("on_epoch_end", self, epoch=epoch, metrics=metrics)
            
        save_dir = os.path.join(self.config.training.save_dir, f"{self.config.task}_{self.config.model.architecture}")
        os.makedirs(save_dir, exist_ok=True)
        best_weights_path = os.path.join(save_dir, "best.pth")
        
        # Save real weights
        self.torch.save({"model_state_dict": self.model.state_dict(), "classes": classes}, best_weights_path)
        print(f"[TimmAdapter] Saved weights to {best_weights_path}")
        
        self.hook_manager.trigger("on_train_end", self)
        
        return {
            "status": "success",
            "save_dir": save_dir,
            "metrics": {"final_loss": epoch_loss, "final_acc": epoch_acc}
        }

    def evaluate(self, data_path: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("TimmAdapter eval pipeline in scaffolding phase.")

    def predict(self, source: str, **kwargs):
        raise NotImplementedError("TimmAdapter predict pipeline in scaffolding phase.")

    def export(self, export_format: str, save_path: str = None, **kwargs) -> str:
        if export_format.lower() != "onnx":
            raise NotImplementedError(f"Export format {export_format} not supported for TimmAdapter yet.")
            
        self.model.eval()
        dummy_input = self.torch.randn(1, 3, 224, 224).to(self.device)
        
        if not save_path:
            base_dir = self.config.training.get("save_dir", "runs")
            arch = self.config.model.get("architecture", "timm_model")
            task = self.config.get("task", "classification")
            save_path = os.path.join(base_dir, f"{task}_{arch}", f"{arch}.onnx")
            
        self.torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=17, # Industry standard for modern deployment
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
        print(f"[TimmAdapter] ONNX Model correctly exported to {save_path}")
        return save_path
