# Version
```zsh
- Python 3.8.0
- torch==1.11.0
- torchvision==0.12.0
- onnx==1.11.0
- onnxruntime==1.11.0
- opset==12
```

# yolov5 models

- [yolov5s.pt](https://drive.google.com/file/d/11O3tat8lioNpRj4YHdd95k38aM9iuOPe/view?usp=sharing)

- [yolov5m.pt](https://drive.google.com/file/d/1uBkFOrScjCSz9779XbkdWZ1WHLacG_8m/view?usp=sharing)

- [yolov5s.onnx](https://drive.google.com/file/d/1Nddq8H-EAIE8Acpc3voMfgjVbZZmv6mf/view?usp=sharing)

- [yolov5m.onnx](https://drive.google.com/file/d/15wMVTwhLTcw1nXvy-2RCCAJq1Vq-2ePe/view?usp=sharing)

# convert to onnx

1. download pytorch model
2. put the model to weight foloder
3. run below commands


```zsh
# create env
$ conda create -n <name> python==3.8
# install pkg
$ pip3 install -r requirements.txt
# convert
$ python3 convert.py --weights weights/yolov5s.pt --include onnx #torchscript
```


# References

- [yolov5](https://github.com/ultralytics/yolov5)
