import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("./footage/cam.mov")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


line_pts = [(0, h//2), (w, h//2)]

speed_obj = solutions.SpeedEstimator(region=line_pts,
                                      model="./runs/detect/train4/weights/best.pt",
                                      show=True,)
org = (50,50)
color = (0,0,255)

while cap.isOpened():
  success, im0 = cap.read()
  if not success:
    break
  results = speed_obj.process(im0)
  #video_writer.write(results.plot_im)
  speed_indic = cv2.putText(im0, str(results.total_tracks), org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
  printed = cv2.imshow('img', speed_indic)
  #print(type(printed))
  video_writer.write(results.plot_im)
  #print(results.total_tracks)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

