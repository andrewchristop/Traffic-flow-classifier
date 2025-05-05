from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import numpy as np

model = YOLO("yolo11n.pt")
confmat = ConfusionMatrix(nc=3)
metrics = model.val(data="./default/coco.yaml", classes=[2,5,7], plots=True)
confmat.plot(normalize=True, save_dir='.', names=("car", "bus", "truck"))
print(metrics.box.map)
