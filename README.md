# Yolo Object Detection on NVIDIA Jetson Nano 

This repository provides a simple and easy process for camera installation, software and hardware setup, and object detection using Yolov5 on NVIDIA Jetson Nano.
The project uses [CSI-Camera](https://github.com/JetsonHacksNano/CSI-Camera) to create a pipeline and receive a frame from the CSI camera, and [Yolov5](https://github.com/ultralytics/yolov5) to detect objects, providing a complete and executable code on the Jetson Development Kits.
Check out [CodePlay jetson nano youtube playlist](https://www.youtube.com/watch?v=5-SIV7r2uiU&list=PLZIi3Od9VUwW49q6T1VjShktoOgrDi3O4) for more. 

## Download Model
[yolov5 models](https://github.com/ultralytics/yolov5/releases)

Download the model using the command below and move to weights folder.
```
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```

## Requirements
[Camera setup guide](https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477/)

[Arducam IMX477 driver](https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477-how-to-install-the-driver/)

[PyTorch & torchvision for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048)

## Inference

Run
```
python3 JetsonYolo.py
```


