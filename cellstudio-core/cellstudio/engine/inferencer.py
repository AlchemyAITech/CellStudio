import time
from typing import Any, Dict, List

import numpy as np


class BaseONNXInferencer:
    """
    Lightweight, high-performance ONNX model inference wrapper.
    Decoupled from heavy training frameworks (PyTorch/Ultralytics).
    Suitable for production deployment.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is not installed. Please `pip install onnxruntime`.")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device in ['cuda', 'gpu'] else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_details = self.session.get_inputs()[0]
        self.output_details = self.session.get_outputs()
        self.input_name = self.input_details.name
        
        self.input_shape = self.input_details.shape
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert raw image to ONNX input tensor format.
        Must return numpy array.
        """
        raise NotImplementedError

    def postprocess(self, outputs: List[np.ndarray]) -> Any:
        """
        Convert raw ONNX output tensors to structured predictions.
        """
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full inference pipeline with millisecond telemetry.
        """
        # 1. Preprocess
        t0 = time.perf_counter()
        input_tensor = self.preprocess(image)
        t1 = time.perf_counter()
        
        # 2. Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        t2 = time.perf_counter()
        
        # 3. Postprocess
        results = self.postprocess(outputs)
        t3 = time.perf_counter()
        
        return {
            "results": results,
            "latency_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "inference": (t2 - t1) * 1000.0,
                "postprocess": (t3 - t2) * 1000.0,
                "total": (t3 - t0) * 1000.0
            }
        }

class ClassificationONNXInferencer(BaseONNXInferencer):
    """Specific implementation for standard classification models (like ResNet)"""
    def __init__(self, model_path: str, device: str = "cpu", target_size=(224, 224)):
        super().__init__(model_path, device)
        self.target_size = target_size
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        # Resize
        img = cv2.resize(image, self.target_size)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Standard ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension NCHW
        img = np.expand_dims(img, axis=0)
        return img
        
    def postprocess(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        # Typically outputs[0] is logits shape (1, num_classes)
        logits = outputs[0][0]
        # Softmax
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        return {
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probs.tolist()
        }
