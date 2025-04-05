import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("./footage/cam2.mov")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


line_pts = [(0, h//2), (w, h//2)]
#line_pts = [(0, 700), (w, 700)]

speed_obj = solutions.SpeedEstimator(region=line_pts,
                                      model="./runs/detect/train4/weights/best.pt",
                                      show=True,)

while cap.isOpened():
  success, im0 = cap.read()
  if not success:
    break
  results = speed_obj(im0)
  #tracks = model.track(im0, persist=True, show=False)
  #im0 = speed_obj.estimate_speed(im0, tracks)
  video_writer.write(results.plot_im)
  #video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

