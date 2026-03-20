import traceback
from cellstudio.models.adapters.ultralytics_adapter import UltralyticsDetAdapter

try:
    print("Attempting to build UltralyticsDetAdapter...")
    model = UltralyticsDetAdapter(yaml_model='yolov8n.pt')
    print("Success!")
except Exception as e:
    print("Failed!")
    traceback.print_exc()
