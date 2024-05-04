
from ultralytics import YOLO

modely = YOLO('yolov8n.pt', verbose=False)

results = modely.track(source="./test_data/walks.mp4", show=True, verbose=False, classes=[0])

