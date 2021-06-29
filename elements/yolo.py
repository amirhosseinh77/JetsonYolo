import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CAR_DETECTION():
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
                if int(p[5]) in self.classes.keys():
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

class PLATE_DETECTION():
    def __init__(self, model_path, classes):
        self.classes = classes
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = 320

    def detect(self, frame, items):
        
        for item in items:
            [(xmin,ymin),(xmax,ymax)] = item['bbox']
            [xmin,xmax] = np.clip([xmin,xmax],0,frame.shape[1])
            [ymin,ymax] = np.clip([ymin,ymax],0,frame.shape[0])
            cropped_image = frame[ymin:ymax, xmin:xmax]
            
            height, width = cropped_image.shape[:2]
            new_height = int((((self.input_width/width)*height)//32)*32)

            img = cv2.resize(cropped_image, (self.input_width,new_height))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = np.moveaxis(img,-1,0)
            img = torch.from_numpy(img).to(device)
            img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.yolo_model(img, augment=False)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None)
            if pred[0] is not None and len(pred):
                try:
                    p = pred[0][0]
                    if int(p[5]) in self.classes.keys():
                        score = np.round(p[4].cpu().detach().numpy(),2)
                        label = self.classes[int(p[5])]
                        p_xmin = int(p[0] * cropped_image.shape[1] /self.input_width) + xmin
                        p_ymin = int(p[1] * cropped_image.shape[0] /new_height) + ymin
                        p_xmax = int(p[2] * cropped_image.shape[1] /self.input_width) + xmin
                        p_ymax = int(p[3] * cropped_image.shape[0] /new_height) + ymin

                    item['plate_bbox'] = [(p_xmin,p_ymin),(p_xmax,p_ymax)]
                except:
                    item['plate_bbox'] = None
        return items

class CHAR_EXTRACTION():
    def __init__(self, model_path, classes):
        self.classes = classes
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = 320

    def detect(self, frame, items):
        
        for item in items:
            if item['plate_bbox'] is not None:
                [(xmin,ymin),(xmax,ymax)] = item['plate_bbox']
                cropped_image = frame[ymin:ymax, xmin:xmax]

                height, width = cropped_image.shape[:2]
                new_height = int((((self.input_width/width)*height)//32)*32)

                img = cv2.resize(cropped_image, (self.input_width,new_height))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = np.moveaxis(img,-1,0)
                img = torch.from_numpy(img).to(device)
                img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                pred = self.yolo_model(img, augment=False)[0]
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None)
                plate_res = []
                if pred[0] is not None and len(pred):    
                    char_preds = pred[0].cpu().detach().numpy()
                    sorted_chars = char_preds[np.argsort(char_preds[:, 0])]
                    for p in sorted_chars:
                        label = self.classes[int(p[5])]
                        if int(p[5]) in self.classes.keys():
                            plate_res.append(label)
                
                item['lp'] = plate_res
            else:
                item['lp'] = None
                    #     score = np.round(p[4].cpu().detach().numpy(),2)
                    #     label = self.classes[int(p[5])]
                    #     p_xmin = int(p[0] * cropped_image.shape[1] /640) + xmin
                    #     p_ymin = int(p[1] * cropped_image.shape[0] /384) + ymin
                    #     p_xmax = int(p[2] * cropped_image.shape[1] /640) + xmin
                    #     p_ymax = int(p[3] * cropped_image.shape[0] /384) + ymin

                    #     item['plate_bbox'] = [(p_xmin,p_ymin),(p_xmax,p_ymax)]
                
        return items
