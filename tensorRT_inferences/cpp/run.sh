# /bin/sh
#### CREATE DATASET ####
#mkdir test_dataset
#python3 unet/prepareData.py --input_image test_images/brain_mri_4947.tif --input_tensor test_dataset/input_0.pb --output_tensor test_dataset/output_0.pb

#### tensorRT ONNX inference
cd code-samples/posts/TensorRT-introduction
./simpleOnnx_1 ../../../unet/unet.onnx ../../../test_dataset/input_0.pb

