import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OBJ_DETECTION():
    def __init__(self, model_path, classes):
        self.classes = classes
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = 320

    def detect(self,main_img):
        height, width = main_img.shape[:2]
        new_height = int((((self.input_width/width)*height)//32)*32)

        img = cv2.resize(main_img, (self.input_width,new_height))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img,-1,0)
        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None)
        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(),2)
                label = self.classes[int(p[5])]
                xmin = int(p[0] * main_img.shape[1] /self.input_width)
                ymin = int(p[1] * main_img.shape[0] /new_height)
                xmax = int(p[2] * main_img.shape[1] /self.input_width)
                ymax = int(p[3] * main_img.shape[0] /new_height)

                item = {'label': label,
                        'bbox' : [(xmin,ymin),(xmax,ymax)],
                        'score': score
                        }

                items.append(item)

        return items
