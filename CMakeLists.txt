CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
cmake_policy(SET CMP0048 NEW)
project(MattingPlugin VERSION "1.0.0.0")


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-pthread")

# Torch
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/libtorch/)
# OpenCV
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/build/)

# MattingPlugin, Test
add_subdirectory(./src/MattingPlugin)
add_subdirectory(./src/Test)