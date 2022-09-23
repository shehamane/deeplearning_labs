cmake_minimum_required(VERSION 3.23)
project(lab1)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_EXE src/main.cpp)
set(SOURCE_LIB
        src/lib/Tensor.cpp src/lib/NeuralNetwork.cpp src/lib/Layer.cpp src/lib/DataSet.cpp)

add_library(LIB STATIC ${SOURCE_LIB})
add_executable(lab1 ${SOURCE_EXE})

target_link_libraries(lab1 LIB)
