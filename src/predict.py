import sys
from ultralytics import YOLO

print("Script started")

model = YOLO("models/best.pt")
source = sys.argv[1] if len(sys.argv) > 1 else "test_media"
print("Predicting on:", source)

results = model.predict(source=source, save=True, conf=0.4, project="outputs", name="pred")
print("Saved to:", results[0].save_dir)