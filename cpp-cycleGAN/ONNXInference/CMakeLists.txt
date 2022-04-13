cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
# torch 
set(LIBTORCH_DIR /tmp/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

# cv2
find_package(OpenCV REQUIRED)
# onnxruntime
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


add_subdirectory(src)
