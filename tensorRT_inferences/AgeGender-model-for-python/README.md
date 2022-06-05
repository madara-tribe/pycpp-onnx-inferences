# yolo_age_gender_model

# Versions
- Ubuntu 18.04
- cudnn7
- cuda  10.1
- python 3.7.0
- keras 2.3.1
- tensorflow 2.3.0


# Network overall
<img src="https://user-images.githubusercontent.com/48679574/115017273-4840a580-9e84-11eb-8fc7-2206d79d8fef.png" width="800px">


# Emphasize parts of age gender model 

## 1. Contrast normalize instead of opencv (Low time cost)

<b>No normalize image</b><hr>

<img src="https://user-images.githubusercontent.com/48679574/114886985-0822e980-9dd6-11eb-909f-c5f48930a0bf.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/114886994-09ecad00-9dd6-11eb-9997-692d06160d1f.png" width="200px">


<b>Contrast normalize instead opencv(mean_avg>=129)</b><hr>
```python
MEAN_AVG = float(130.509485819935)
def to_mean_pixel(img, avg):
    return (img - 128)*(128/avg)
```

<img src="https://user-images.githubusercontent.com/48679574/114887032-1113bb00-9dd6-11eb-9bfe-b7805b9e0370.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/114887040-12dd7e80-9dd6-11eb-8913-09959a2c5ed9.png" width="200px">


<b>Opencv equalizeHist</b><hr>
```python
def equalizeHist(img):
    for j in range(3):
        img[:, :, j] = cv2.equalizeHist(img[:, :, j])
    return img
```

<img src="https://user-images.githubusercontent.com/48679574/114887071-1a048c80-9dd6-11eb-8005-8dd90868f631.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/114887078-1bce5000-9dd6-11eb-96f1-5e70799520ed.png" width="200px">



## 2.Age gender model performance

<b>loss curve</b>.              
<img src="https://user-images.githubusercontent.com/48679574/119448282-0b61ab80-bd6c-11eb-8fb7-699a65e65704.png" width="400px">




## 3.Result images of yolo_age_gender_model

<b>image1.jpg</b>.： <b>prediction1.png(age : 21, gender : Female)</b>　　　<b>image2.jpg</b>.  <b>prediction2.png(age : 2, gender : Female)</b>

<img src="https://user-images.githubusercontent.com/48679574/119312320-30431980-bcad-11eb-933a-d9f5fa7a84ea.jpg" width="200px"><img src="https://user-images.githubusercontent.com/48679574/119312326-320cdd00-bcad-11eb-9b29-6d5315b82547.png" width="200px">　　<img src="https://user-images.githubusercontent.com/48679574/119312349-389b5480-bcad-11eb-8f04-f02b7122578d.jpg" width="200px"><img src="https://user-images.githubusercontent.com/48679574/119312355-3afdae80-bcad-11eb-88f3-c15ec6036618.png" width="200px">






## 4. generation and Identity age prediction for complicated real age prediction

The model predict 6 generations and 21 Identity(approximately) age. Human has identity that influence thier face appearance, so to predict approximately identity age is better way to estimate real age. Because it reduces mistaken prediction.

This model is to calculate real age with 6 generations and 21 Identity(approximately) age.

<img src="https://user-images.githubusercontent.com/48679574/119289532-bcd9e180-bc85-11eb-8a82-0ae712936c5a.png" width="700px">





## 5. loss weights when compile

age gender loss are MSE loss and binary_cross_entropy loss. Loss weight when model compile is as follows:

```python
model.compile(optimizer=adam, 
                  loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
                  loss_weights={'age_output': 0.25, 'gender_output': 10},
                  metrics={'age_output': 'mse', 'gender_output': 'accuracy'})
```
- [Keras loss weights](https://stackoverflow.com/questions/48620057/keras-loss-weights)




# Convert age-gender-model to onnx

- python 3.7.0
- tensorflow 2.3.0
- keras 2.3.1
- keras2onnx 1.8.0
- onnxconverter-common 1.6.0
- onnx 1.6.0


# References

- [yu4u/age-gender-estimation](https://github.com/yu4u/age-gender-estimation)
- [UTKFace dataset](https://susanqq.github.io/UTKFace/)

