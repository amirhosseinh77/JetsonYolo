import cv2
import numpy as np
from elements.yolo import CAR_DETECTION as Person_Detection

Person_classes = {0: 'Person'}
Person_Detector = Person_Detection('weights/car_model.pt', Person_classes)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            # detection process
            Persons = Person_Detector.detect(frame)

            # plotting
            for Person in Persons:
                print(Person)
                label = Person['label']
                score = Person['score']
                [(xmin,ymin),(xmax,ymax)] = Person['bbox']
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), [0,255,255] , 2) 
                # frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, [0,255,255], 2, cv2.LINE_AA)

        cv2.imshow("CSI Camera", frame)
        keyCode = cv2.waitKey(30)
        if keyCode == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")
