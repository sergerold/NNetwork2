cmake_minimum_required(VERSION 3.24)
project(NNetwork2)

set(CMAKE_CXX_STANDARD 17)

add_executable(NNetwork2 main.cpp NLayer.cpp NLayer.h NNetwork.cpp NNetwork.h Training.cpp Training.h Debug.cpp Debug.h)
