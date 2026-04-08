from cellpose import models
import time

print("Initializing CellposeModel...")
t0 = time.time()
model = models.CellposeModel(model_type='cyto3', gpu=True)
print(f"Initialization took {time.time()-t0:.2f}s!")
