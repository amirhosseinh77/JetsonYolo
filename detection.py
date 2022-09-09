import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION

def warp_coords(xy: np.float32, matrix) -> np.float32:
        return cv2.warpPerspective(xy, matrix, (800, 600), flags=cv2.INTER_LINEAR)
        return np.matmul(xy, matrix)

def get_center_coords(obj) -> list:
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        return np.float32([(xmax - xmin) / 2, (ymax - ymin) / 2])

def push_coords(coords: np.float32) -> None:
        print(coords)

VIDEO_SIZE = (1280, 720)

Object_classes = ['cig_butt']
Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/best.pt', Object_classes)

# get perspective transformation (calibrate camera)
input_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
output_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
warp_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

cap = cv2.VideoCapture(0)
if cap.isOpened():
        while True:
                ret, frame = cap.read()
                frame = cv2.resize(frame, VIDEO_SIZE, fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
                frame = cv2.warpPerspective(frame, warp_matrix, VIDEO_SIZE, flags=cv2.INTER_CUBIC)
                if ret:
                        objs = Object_detector.detect(frame)
                        for obj in objs:
                                center_coords = get_center_coords(obj)
                                #warped_coords = warp_coords(center_coords, warp_matrix)
                                push_coords(center_coords)
                                #push_coords(warped_coords)
                        else:
                                None
        cap.release()
else:
        print("Unable to open camera")
