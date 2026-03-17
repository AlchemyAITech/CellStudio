import os
import numpy as np
from typing import List, Dict, Any, Union

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class ONNXInferencer:
    """
    Lightweight, dependency-free (except ONNXRuntime and Numpy) inferencer 
    for deploying PathStudio models into production environments without 
    heavy training framework payloads (like PyTorch or Ultralytics).
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        if ort is None:
            raise ImportError("onnxruntime is required for deployment. Please `pip install onnxruntime`")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
        self.model_path = model_path
        self.device = device
        
        # Configure execution providers
        providers = ['CPUExecutionProvider']
        if device == "cuda" or device == "gpu":
            providers = ['CUDAExecutionProvider'] + providers
            
        sess_options = ort.SessionOptions()
        # You can add optimizations here
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        except Exception as e:
            # Fallback to CPU if CUDA provider errors out on mismatch
             print(f"[Warning] Failed to load provider {providers}. Falling back to CPU. Error: {e}")
             self.session = ort.InferenceSession(self.model_path, sess_options, providers=['CPUExecutionProvider'])

        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        
        # Typically one input for vision models, but we prepare for flex
        self.input_names = [i.name for i in self.input_details]
        self.output_names = [o.name for o in self.output_details]
        
    def _preprocess(self, image_bgr: np.ndarray, target_size=(640, 640)) -> np.ndarray:
        """
        Generic preprocessing. Convert BGR to RGB, resize, normalize, and transpose to NCHW.
        Note: Specific tasks might require overriding this method.
        """
        import cv2 # Local import for deployment module
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Basic resize (without letterboxing for simplicity in this baseline)
        img = cv2.resize(img, target_size)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add N dim: NCHW
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Run inference on a single image.
        Returns the raw model output which should be decoded by task-specific postprocessors.
        """
        import cv2
        
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at {image}")
            img_bgr = cv2.imread(image)
        else:
            img_bgr = image
            
        # Determine expected shape from ONNX graph 
        # e.g., ['batch', 'channels', 'height', 'width'] => [-1, 3, 640, 640]
        input_shape = self.input_details[0].shape
        # Safety check if dynamic dimensions are used
        h, w = 640, 640
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
             h, w = input_shape[2], input_shape[3]
             
        tensor = self._preprocess(img_bgr, target_size=(w, h))
        
        # Execute
        raw_outputs = self.session.run(self.output_names, {self.input_names[0]: tensor})
        
        # Wrap outputs mapped to output names
        result = {name: out for name, out in zip(self.output_names, raw_outputs)}
        return result
