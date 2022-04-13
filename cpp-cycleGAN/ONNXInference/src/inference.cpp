#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#define W 256
#define H 256
#define ONNX_MODEL_PATH "../../data/cycleGAN_AB.onnx"
#define IMG_PATH "../../data/hourse.jpg"

using namespace cv;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

void image_info(cv::Mat image){
    double min, max;
    std::cout << "size: " << image.size() << "  dims: " << image.dims << ", channels: " << image.channels() << std::endl;
    cv::minMaxLoc(image, &min, &max);
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
}

cv::Mat normalization(std::string imageFilepath){
    cv::Mat CHWImage;
    cv::Mat image = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::resize(image, image, cv::Size(H, W),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(image, image,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0/127.5, -1);
    
    image_info(image);
    // HWC to CHW
    cv::dnn::blobFromImage(image, CHWImage);
    std::cout << "dims = " << CHWImage.dims << "size: " << CHWImage.size() << std::endl;
    return CHWImage;
}

   
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
    image_info(segMat);
    imwrite("GANoutput.png", segMat);
}
    
int onnx_predict(bool useCUDA){
    // Define data and model
    std::string instanceName{"cycleGAN inference"};
    std::string modelFilepath{ONNX_MODEL_PATH};
    std::string imageFilepath{IMG_PATH};
    
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
    OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    }

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    //start inference session
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    
    const char* inputName = session.GetInputName(0, allocator);
    //Input Name: input1
    
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    // Input Type: float
    
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    // Input Dimensions: [1, 3, 256, 256]
    
    const char* outputName = session.GetOutputName(0, allocator);
    // Output Name: output1
    
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    // Output Type: float
    
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    // Output Dimensions: [1, 3, 256, 256]
    float output_size = 3 * H * W;
        
    // Normalize iamge
    cv::Mat preprocessedImage = normalization(imageFilepath);

    
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());
    
    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
              output_size == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);
    
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    std::vector<Ort::Value> output_tensors;
    
    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
    
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), (size_t)inputNames.size(), outputNames.data(), outputTensors.data(), (size_t)outputNames.size());
    std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
    std::cout << "ONNX Inference Latency: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                           begin)
                             .count() / static_cast<float>(numTests)
                  << " ms" << std::endl;
    std::cout << "outputTensorValues: "<< outputTensorValues.size() << std::endl;
    PostProc(outputTensorValues);
    return 0;
}


int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    return onnx_predict(useCUDA);
}


