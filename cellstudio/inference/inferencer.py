import torch
from typing import Dict, Any

from ..engine.config.config import Config
from ..models.builder import MODEL_REGISTRY
from ..pipeline.compose import Compose

class CellStudioInferencer:
    """
    Decoupled Inferencer Core for CellStudio algorithms.
    Provides identical inference streams for API endpoints and CLI tools.
    """
    def __init__(self, config_path: str, weight_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cfg = Config.fromfile(config_path)
        
        # 1. Build Model Architecture
        model_cfg = self.cfg.get('model')
        if not model_cfg:
            raise KeyError("Configuration MUST contain a 'model' dictionary.")
            
        self.model = MODEL_REGISTRY.build(model_cfg)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Inject Pre-Trained Weights
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        # Adapt for models loaded under standard dicts vs DDP dicts
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        self.model.load_state_dict(state_dict, strict=False)
        print(f"[Inferencer] Attached pre-trained weights from {weight_path}")
        
        # 3. Assemble Inference Pipeline matching Val Set
        val_loader_cfg = self.cfg.get('val_dataloader', {})
        dataset_cfg = val_loader_cfg.get('dataset', {})
        pipeline_cfg = dataset_cfg.get('pipeline', [])
        
        if not pipeline_cfg:
            raise ValueError("Configuration MUST define a valid dataset pipeline under val_dataloader.")
            
        self.pipeline = Compose(pipeline_cfg)

    @torch.no_grad()
    def __call__(self, img_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a solitary filepath.
        """
        # Pack fake dict matching DataSample schema
        data = dict(img_path=str(img_path))
        
        # Sequentially map through the pipeline nodes
        data = self.pipeline(data)
        
        # Prepare inputs (add batch dimension)
        inputs = data['imgs'].unsqueeze(0).to(self.device)
        
        # Forward pass bypassing gradients securely
        if hasattr(self.model, 'forward_test'):
            outputs = self.model.forward_test(inputs)
        else:
            outputs = self.model(inputs)
            
        # Parse logic specific to classification output schemas
        # Ex: {'probs': tensor([1, 2]), 'preds': tensor([0])}
        result = {}
        if isinstance(outputs, dict) and 'probs' in outputs and 'preds' in outputs:
            probs = outputs['probs'][0].cpu().numpy().tolist()
            pred = int(outputs['preds'][0].item())
            result['class_id'] = pred
            result['confidence'] = probs[pred]
            result['probabilities'] = probs
            
        return result
