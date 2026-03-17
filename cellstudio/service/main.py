import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from pathstudio.service.schemas import InferenceResponse, TaskResponse, BBoxResponse
from pathstudio.deploy.inferencer import ONNXInferencer

app = FastAPI(title="PathStudio API Service", version="1.0.0")

# Allow Web UI origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for loaded inferencers
# In production, this might be handled by Triton Inference Server or specialized model servers.
loaded_models = {}

@app.get("/api/v1/health")
def health_check():
    """Service health check endpoint."""
    return {"status": "up", "models_loaded": list(loaded_models.keys())}


@app.post("/api/v1/predict", response_model=InferenceResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_path: str = Form(...),
    task_type: str = Form("detect"),
    device: str = Form("cpu")
):
    """
    Execute ONNX model inference on an uploaded image.
    """
    if cv2 is None:
        raise HTTPException(status_code=500, detail="OpenCV required for image processing in the backend. `pip install opencv-python-headless`")
        
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # Load Model (cache it if seen before)
    if model_path not in loaded_models:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model path not found: {model_path}")
        try:
            inferencer = ONNXInferencer(model_path=model_path, device=device)
            loaded_models[model_path] = inferencer
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {str(e)}")
            
    inferencer = loaded_models[model_path]
    
    try:
        raw_results = inferencer.predict(image_bgr)
        
        # Here we would have task-specific decoders.
        # For MVP, we pass the raw shape / dummy boxes back to demonstrate parsing flow.
        response = InferenceResponse(
            status="success",
            task_type=task_type,
            metadata={"raw_output_keys": list(raw_results.keys())}
        )
        
        # Example dummy decoder for YOLOv8 detection structure
        if task_type == "detect":
            # Just simulating returning a bounding box based on successful execution
            response.bboxes.append(
                BBoxResponse(xmin=10.0, ymin=10.0, xmax=50.0, ymax=50.0, label="cell", confidence=0.99)
            )
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


def _background_train_task(task_type: str, dataset_schema_path: str, save_dir: str):
    """
    Simulated background task executor. 
    In Phase 3/4, this will invoke PathStudio Engine via CLI or Python API 
    and push progress to Redis/Celery.
    """
    print(f"[Celery/Background Worker] Starting training task: {task_type}")
    print(f"[Celery/Background Worker] Dataset: {dataset_schema_path}")
    print(f"[Celery/Background Worker] Models will be saved to: {save_dir}")
    # engine = load_engine_from_config(dataset_schema_path)
    # engine.train()


@app.post("/api/v1/train", response_model=TaskResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    task_type: str = Form(...),
    dataset_schema_path: str = Form(...),
    save_dir: str = Form("./outputs/api_train_runs")
):
    """
    Initiate an asynchronous training pipeline.
    """
    if not os.path.exists(dataset_schema_path):
         raise HTTPException(status_code=404, detail=f"Dataset JSON Schema not found: {dataset_schema_path}")
         
    import uuid
    task_id = str(uuid.uuid4())
    
    # Enqueue task
    background_tasks.add_task(_background_train_task, task_type, dataset_schema_path, save_dir)
    
    return TaskResponse(
        task_id=task_id, 
        status="pending", 
        message="Training task queued successfully"
    )
