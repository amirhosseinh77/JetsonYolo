import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION

def warp_coords(xy: list, matrix) -> list:
        return v2.warpPerspective(xy, matrix, (800, 600), flags=cv2.INTER_LINEAR)

def get_center_coords(obj) -> list:
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        return ((xmax - xmin) / 2, (ymax - ymin) / 2)

def push_coords():
        ...

Object_classes = ['cig_butt']
Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/best.pt', Object_classes)

# get perspective transformation (calibrate camera)
input_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
output_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
warp_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

cap = cv2.VideoCapture(0)
if cap.isOpened():
        window_handle = cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("USB Camera", 0) >= 0:
                ret, frame = cap.read()
                if ret:
                        # detection process
                        objs = Object_detector.detect(frame)

                        # plotting
                        for obj in objs:
                                center_coords = get_center_coords(obj)
                                warped_coords = warp_coords(center_coords, warp_matrix)

                cv2.imshow("USB Camera", frame)
                keyCode = cv2.waitKey(30)
                if keyCode == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()
else:
        print("Unable to open camera")
