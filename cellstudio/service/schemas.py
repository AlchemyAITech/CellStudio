from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class BBoxResponse(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str
    confidence: float

class PolygonResponse(BaseModel):
    points: List[float]
    label: str
    confidence: Optional[float] = None

class InferenceResponse(BaseModel):
    status: str
    message: str = ""
    task_type: str
    bboxes: List[BBoxResponse] = []
    polygons: List[PolygonResponse] = []
    cls_labels: List[Dict[str, float]] = [] # e.g. [{"benign": 0.98}, {"malignant": 0.02}]
    metadata: Dict[str, Any] = {}

class TrainRequest(BaseModel):
    task: str # detect, classify, segment
    dataset_schema_path: str
    config_overrides: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
