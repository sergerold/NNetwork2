cmake_minimum_required(VERSION 3.24)
project(NNetwork2)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-fopenmp -O3 -funroll-loops -march=native -Wall -Wextra -Wshadow -Wconversion -Wpedantic")

add_executable(NNetwork2 main.cpp NLayer.cpp NLayer.h NNetwork.cpp NNetwork.h Training.cpp Training.h Debug.cpp Debug.h Data.cpp Data.h DataSpecs.h)
