# Versions
```bash
・ cuda 10.1
・ nvidia driver 430.64
・ Python 3.7.11
・ pytorch 1.7.1+cu101
・ torchvision 0.8.2+cu101
・ onnx 1.10.2
・ onnxruntime 1.9.0
・ PySide6 6.2.0
```

# abstract about cycleGAN
cycle GAN mainly training horse to zebra and zebra to horse.
<img src="https://user-images.githubusercontent.com/48679574/142752809-9243c8bd-e0bb-4d5d-9798-4a9f4181c85f.png" width="650px">





## pytorch code for speed up
```python
# at data loder
num_workers = 8 if os.cpu_count() > 8 else os.cpu_count()
pin_memory = True
DataLoader(〜〜, num_workers=num_workers, pin_memory=pin_memory)
# before training
torch.backends.cudnn.benchmark =True
```

# Result
## 1. cycleGAN with pyside ML GUI app on Mac(CPU)
```pythonn
$ python3 Qtapp.py
```

![qtapp](https://user-images.githubusercontent.com/48679574/142753049-7aa84817-1f04-4301-8b2c-62a19d844745.gif)


## 2. result : convrting hourse to zebra as torch and onnx
training images is 1139 test image is 150
<b>at 199 epoch</b>

<img src="https://user-images.githubusercontent.com/48679574/142752812-2606162d-2cdb-419b-b6e0-b2d07def95f0.jpg" width="300px"><img src="https://user-images.githubusercontent.com/48679574/142752813-9d69f009-a598-4f1b-8bac-efe908bc392e.png" width="300px">


## 3. Prediction speed with ONNX format on Cuda
<b>pytorch inference speed</b>

```Inference Latency (milliseconds) is 8438.8799 [ms]```

<b>onnx model inference speed</b>

```Inference Latency (milliseconds) is 45.3539218902588 [ms]```

<img src="https://user-images.githubusercontent.com/48679574/142753020-b867513f-3c0e-4b3d-a75a-9b28fcc17407.png" width="400px">


## 4. Generator and Discriminator loss curve

<img src="https://user-images.githubusercontent.com/48679574/142752865-7a962b27-5c90-4d62-a44c-d36d3328e9b9.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/142752867-4d6a39bd-b919-4bdb-8ece-e5b1b12ea639.png" width="200px">



# References
## Pyside References
- [ImageEditor_PyQt](https://github.com/koharite/ImageEditor_PyQt)
- [QMainWindow](http://blawat2015.no-ip.com/~mieki256/diary/201610161.html)
- [pyside Widget introductions](https://dftalk.jp/?p=20768)

## Pytorch
- [Tensorflow GPU, CUDA, CuDNNのバージョン早見表](https://qiita.com/chin_self_driving_car/items/f00af2dbd022b65c9068)
- [PyTorchでの学習・推論を高速化するコツ集](https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587)
