import os
from omegaconf import DictConfig
from typing import Dict, Any

from cellstudio.backends.base.adapter import BaseBackendAdapter
from cellstudio.datasets.schema import CellDatasetConfig

class TimmAdapter(BaseBackendAdapter):
    """
    Adapter for PyTorch Image Models (timm).
    Provides classification training, evaluation, and export capabilities 
    conforming strictly to the CellStudio BaseBackendAdapter protocol.
    """
    def __init__(self, config: DictConfig, device: str = None):
        try:
            import timm
            import torch
            from torch import nn
            self.timm = timm
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("PyTorch and timm libraries are required for TimmAdapter. Please `pip install torch torchvision timm`")
            
        # The engine layer passed down the device. If None, fallback to config.
        device_str = device if device else config.env.get("device", "cpu")
        self.device = self.torch.device(device_str)
        super().__init__(config, device=self.device)

    def _build_model(self):
        """Initializes TIMM model based on architecture defined in config."""
        model_arch = self.config.model.get("architecture", "resnet50")
        pretrained = self.config.model.get("pretrained", True)
        weights_path = self.config.model.get("pretrained_weights", None)
        
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
        from PIL import Image
        from cellstudio.datasets.transforms.factory import build_transforms
        
        class CellJsonDataset(Dataset):
            def __init__(self, items, class_list, transform=None):
                self.items = items
                self.class_list = class_list
                self.transform = transform
                
            def __len__(self): return len(self.items)
            def __getitem__(self, idx):
                item = self.items[idx]
                img = Image.open(item.image_path).convert('RGB')
                if self.transform: img = self.transform(img)
                return img, self.class_list.index(item.cls_labels[0])
                
        # Parse Data Augmentation
        aug_configs = self.config.data.get("augmentations", [])
        train_transform = build_transforms(aug_configs)
        val_transform = build_transforms([]) # Pure evaluation scale
        
        train_dataset = CellJsonDataset(dataset_config.items, classes, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.config.data.batch_size, shuffle=True, num_workers=0)
        
        val_path = self.config.data.get("val_path", data_path)
        val_dataset_config = CellDatasetConfig.load(val_path)
        val_dataset = CellJsonDataset(val_dataset_config.items, classes, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=self.config.data.batch_size, shuffle=False, num_workers=0)
        
        # Advance Hyperparameter Instantiation (Optimizer, Scheduler, Loss)
        opt_cfg = self.config.training.get("optimizer", {"name": "AdamW", "lr": self.config.training.get("learning_rate", 1e-4)})
        opt_name = opt_cfg.get("name", "AdamW")
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-5))
        mom = float(opt_cfg.get("momentum", 0.9))
        
        if opt_name == "AdamW": optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "SGD": optimizer = self.torch.optim.SGD(self.model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        else: optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        
        sched_cfg = self.config.training.get("scheduler", None)
        scheduler = None
        if sched_cfg and sched_cfg.get("name") == "CosineAnnealingLR":
            scheduler = self.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_cfg.get("T_max", self.config.training.epochs))
            
        loss_cfg = self.config.training.get("loss", {"name": "CrossEntropyLoss"})
        if loss_cfg.get("name") == "CrossEntropyLoss":
            criterion = self.nn.CrossEntropyLoss(label_smoothing=loss_cfg.get("label_smoothing", 0.0))
        else:
            criterion = self.nn.CrossEntropyLoss()
            
        print(f"[TimmAdapter] Starting Expert-Level training for {self.config.training.epochs} epochs")
        
        import pandas as pd
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_val_acc = 0.0
        save_dir = os.path.join(self.config.training.save_dir, f"{self.config.task}_{self.config.model.get('architecture', 'timm')}")
        os.makedirs(save_dir, exist_ok=True)
        best_weights_path = os.path.join(save_dir, "best.pth")
        
        self.hook_manager.trigger("on_train_begin", self)
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0); correct += predicted.eq(targets).sum().item()
                self.hook_manager.trigger("on_train_step_end", self, step=step, loss=loss.item())
                
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total
            
            # --- Per-Epoch Validation Pass ---
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with self.torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0); val_correct += predicted.eq(targets).sum().item()
                    
            epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            epoch_val_acc = val_correct / val_total if val_total > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"[Epoch {epoch}/{self.config.training.epochs}] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}")
            
            history['epoch'].append(epoch); history['train_loss'].append(epoch_train_loss); history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss); history['val_acc'].append(epoch_val_acc); history['lr'].append(current_lr)
            
            if scheduler: scheduler.step()
            
            # Drop checkpoint
            if epoch_val_acc >= best_val_acc:
                best_val_acc = epoch_val_acc
                self.torch.save({"model_state_dict": self.model.state_dict(), "classes": classes}, best_weights_path)
                
            self.hook_manager.trigger("on_epoch_end", self, epoch=epoch, metrics={"loss": epoch_train_loss, "val_acc": epoch_val_acc})
            
        # Export Telemetry CSV
        csv_path = os.path.join(save_dir, "training_history.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"[TimmAdapter] Persisted training logs to {csv_path}")
        print(f"[TimmAdapter] Expert Training completed. Saved weights to {best_weights_path}")
        
        self.hook_manager.trigger("on_train_end", self)
        
        return {
            "status": "success",
            "save_dir": save_dir,
            "metrics": {"final_train_loss": epoch_train_loss, "best_val_acc": best_val_acc}
        }

    def evaluate(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Executes PyTorch/Timm evaluation loop.
        Returns standard metrics and raw predictions for ROC/PR curves.
        """
        if not data_path.endswith('.json'):
            raise ValueError(f"TimmAdapter expects a standard PathStudio .json schema, got {data_path}")
            
        print(f"[TimmAdapter] Loading test dataset from {data_path}")
        dataset_config = CellDatasetConfig.load(data_path)
        classes = dataset_config.classes
        
        # Load specific weights if running purely as an evaluator without preceding train loop
        weights_path = self.config.model.get("pretrained_weights")
        if weights_path and os.path.exists(weights_path):
            print(f"[TimmAdapter] Loading weights from {weights_path} for evaluation")
            state_dict_payload = self.torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in state_dict_payload:
                self.model.load_state_dict(state_dict_payload['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict_payload)
            if 'classes' in state_dict_payload:
                classes = state_dict_payload['classes']
                
        # Adjust head if classes changed somehow
        if len(classes) != self.model.num_classes:
            self.model.reset_classifier(len(classes))
            self.model = self.model.to(self.device)
            
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
                label_idx = self.class_list.index(item.cls_labels[0])
                return img, label_idx
                
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = CellJsonDataset(dataset_config.items, classes, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.config.data.batch_size, shuffle=False, num_workers=0)
        
        self.model.eval()
        all_targets = []
        all_preds = []
        all_probs = []
        
        import torch.nn.functional as F
        
        print(f"[TimmAdapter] Starting evaluation on {len(test_dataset)} samples...")
        with self.torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Calculate base metrics
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"[TimmAdapter] Evaluation Results: Acc {acc:.4f} | F1 {f1:.4f}")
        
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "raw_targets": all_targets.tolist(),
            "raw_preds": all_preds.tolist(),
            "raw_probs": all_probs.tolist(),
            "classes": classes
        }

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
