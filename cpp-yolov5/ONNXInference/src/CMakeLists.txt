cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(yolov5s 
	       main.cpp
               detector.cpp
               utils.cpp)

# onnxruntime
target_include_directories(yolov5s PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
target_link_libraries(yolov5s PRIVATE ${ONNX_RUNTIME_LIB})

# cv2
target_include_directories(yolov5s PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov5s PUBLIC ${OpenCV_LIBS})
