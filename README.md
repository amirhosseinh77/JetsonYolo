# Yolo Object Detection on NVIDIA Jetson Nano 

[CodePlay Jetson Nano Playlist](https://www.youtube.com/watch?v=5-SIV7r2uiU&list=PLZIi3Od9VUwW49q6T1VjShktoOgrDi3O4)

## Download Model
[yolov5 models](https://github.com/ultralytics/yolov5/releases)

Download the model using the command below and move to to weights folder.
```
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```

## Requirements
[Camera setup guide](https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477/)

[Arducam IMX477 driver](https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477-how-to-install-the-driver/)

[PyTorch & torchvision for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048)

## Inference
[CSI Camera](https://github.com/JetsonHacksNano/CSI-Camera)

[Yolov5](https://github.com/ultralytics/yolov5)

Run
```
python3 JetsonYolo.py
```


