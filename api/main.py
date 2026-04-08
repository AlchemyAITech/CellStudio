import os
import sys
import yaml
import tempfile
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Inject CellStudio into paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.inference.inferencer import CellStudioInferencer

app = FastAPI(
    title="CellStudio MLOps API",
    description="Full lifecycle (Train, Evaluate, Infer) Industrial Deep Learning Server endpoints.",
    version="2.0.0"
)

INFERENCERS = {}
JOBS = {}  # In-memory tracking dict mapping UUIDs to subprocess.Popen objects

class JobRequest(BaseModel):
    config_yaml: str
    checkpoint_path: str = None  # Expected for eval ops

@app.on_event("startup")
async def startup_event():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
            models = cfg.get('models', {})
            print(f"========== CellStudio API BOOT ==========")
            for target_id, m_cfg in models.items():
                print(f"[API] WARMING CACHE: '{target_id}' ...")
                abs_conf = os.path.join(os.path.dirname(__file__), m_cfg['config'])
                abs_ckpt = os.path.join(os.path.dirname(__file__), m_cfg['checkpoint'])
                try:
                    inst = CellStudioInferencer(abs_conf, abs_ckpt, m_cfg.get('device', 'cuda'))
                    INFERENCERS[target_id] = inst
                    print(f"      [OK] Booted '{target_id}' successfully.")
                except Exception as e:
                    print(f"      [FAILED] '{target_id}' init crash: {e}")
            print(f"=========================================")

@app.post("/predict/{model_id}")
async def predict(model_id: str, file: UploadFile = File(...)):
    """Run Sub-Millisecond inference using memory-warmed VRAM Cache."""
    if model_id not in INFERENCERS:
        raise HTTPException(status_code=404, detail="Model ID not fully warmed up.")
    try:
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = INFERENCERS[model_id](tmp_path)
        os.remove(tmp_path)
        return {"status": "success", "model": model_id, "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train(req: JobRequest):
    """
    Spawns an asynchronous CLI daemon via tools/cli.py train.
    Requires raw YAML config text.
    """
    job_id = str(uuid.uuid4())
    log_file = os.path.join('work_dirs', f'job_{job_id}.log')
    tmp_cfg_path = os.path.join(tempfile.gettempdir(), f'cfg_{job_id}.yaml')
    
    with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
        f.write(req.config_yaml)
        
    cmd = [sys.executable, "-u", "tools/cli.py", "train", tmp_cfg_path]
    os.makedirs('work_dirs', exist_ok=True)
    with open(log_file, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    
    JOBS[job_id] = {'process': proc, 'log_file': log_file, 'type': 'train'}
    return {"status": "dispatched", "job_id": job_id, "config_file": tmp_cfg_path}

@app.post("/evaluate")
async def evaluate(req: JobRequest):
    """
    Spawns an asynchronous evaluation metrics job.
    """
    if not req.checkpoint_path:
        raise HTTPException(status_code=400, detail="evaluate endpoint requires checkpoint_path string")
        
    job_id = str(uuid.uuid4())
    log_file = os.path.join('work_dirs', f'job_{job_id}.log')
    tmp_cfg_path = os.path.join(tempfile.gettempdir(), f'cfg_{job_id}.yaml')
    
    with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
        f.write(req.config_yaml)
        
    cmd = [sys.executable, "-u", "tools/cli.py", "eval", tmp_cfg_path, req.checkpoint_path]
    os.makedirs('work_dirs', exist_ok=True)
    with open(log_file, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    
    JOBS[job_id] = {'process': proc, 'log_file': log_file, 'type': 'eval'}
    return {"status": "dispatched", "job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll pipeline daemon status and tail logs."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job UUID completely unknown")
        
    job = JOBS[job_id]
    proc = job['process']
    ret = proc.poll()
    
    logs = ""
    if os.path.exists(job['log_file']):
        with open(job['log_file'], 'r') as f:
            lines = f.readlines()
            logs = "".join(lines[-50:])  # Head tail last 50 lines to prevent memory explosion
            
    if ret is None:
        state = "RUNNING"
    elif ret == 0:
        state = "SUCCESS"
    else:
        state = "FAILED"
        
    return {"job_id": job_id, "state": state, "exit_code": ret, "tail_logs": logs}

if __name__ == "__main__":
    import uvicorn
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    s_cfg = cfg.get('server', {})
    uvicorn.run("main:app", host=s_cfg.get('host', '0.0.0.0'), port=s_cfg.get('port', 18080), reload=False)
