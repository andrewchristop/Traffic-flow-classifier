from ultralytics import YOLO

model = YOLO("yolo11n.pt")
metrics = model.val(data="./default/coco.yaml", classes=[2,5,7], plots=True)
print(metrics.box.map)
