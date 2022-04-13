cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
# torch 
set(LIBTORCH_DIR /tmp/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

add_executable(inference inference.cpp)
# torch 
target_link_libraries(inference PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET inference PROPERTY CXX_STANDARD 17)

# onnxruntime
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB})

# cv2
target_include_directories(inference PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PUBLIC ${OpenCV_LIBS})
