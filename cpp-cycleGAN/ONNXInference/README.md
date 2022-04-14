# 1. cpp onnx inference with pytorch cpp API

```zsh
make cudarun
make cudain
./run.sh
```

<b>input / output</b>

<img src="https://user-images.githubusercontent.com/48679574/163290432-501dc6f8-f7c7-4f57-9ae0-e7ea89ad32d3.jpg" width="300px"><img src="https://user-images.githubusercontent.com/48679574/163290436-0fe05b87-6d19-415d-a562-ea2987ef86cb.png" width="300px">



# 2. cpp onnx inference latency
<b>cpp onnx inference latency</b>
```
ONNX Inference Latency: 1.84 ms
outputTensorValues: 196608
```

<b>python onnx inference latency</b> 
```
Inference Latency (milliseconds) is 45.3539218902588 [ms]
```

- [python onnx inference](https://github.com/madara-tribe/qt6-onnxed-CycleGan)




# 3. preprocess and postprocess 

## pytorch CHW input 
```cv::dnn::blobFromImage(image, CHWImage);```

## Post Process from shape(1, 256×256×3)
```cpp
void PostProc(std::vector<float> outputTensorValues){
    Mat segMat(H, W, CV_8UC3);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int i = row * W + col;
            float r = outputTensorValues.at(i) * 127.5 + 127.5;
            float g = outputTensorValues.at(i+1) * 127.5 + 127.5;
            float b = outputTensorValues.at(i+2) * 127.5 + 127.5;
            segMat.at<Vec3b>(row, col) = Vec3b(g, r, b);
        }
    }
    //cvtColor(segMat, segMat, COLOR_BGR2RGB);
    imwrite("GANoutput.png", segMat);
}
```


# 4. pytorch Cpp API Reference
- [PyTorch C++（LibTorch）環境構築](https://qiita.com/koba-jon/items/2b15865f5b4c0c9fbbf7)
- [LibTorch(PyTorch C++API)の使い方](https://tokumini.hatenablog.com/entry/2019/08/24/100000)
- [custom_op_test.cc](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/ort_custom_op/custom_op_test.cc)
