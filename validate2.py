from ultralytics import YOLO

model = YOLO("./runs/detect/train4/weights/best.pt")
metrics = model.val(data="./cctv_dataset/data.yaml", plots=True)
print(metrics.box.map)
