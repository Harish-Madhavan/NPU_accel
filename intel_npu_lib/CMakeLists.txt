cmake_minimum_required(VERSION 3.10)
project(intel_npu_acceleration)

set(CMAKE_CXX_STANDARD 17)

# Find PyTorch (LibTorch)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")
find_package(Torch REQUIRED)

# Find OpenVINO
# Ensure OpenCV/OpenVINO paths are in CMAKE_PREFIX_PATH or ENV
find_package(OpenVINO REQUIRED)

# Define source files
set(SOURCES
    csrc/bindings.cpp
    csrc/device.cpp
    csrc/ops.cpp
)

# Create the library
add_library(intel_npu_lib SHARED ${SOURCES})

# Link PyTorch and OpenVINO
target_link_libraries(intel_npu_lib "${TORCH_LIBRARIES}" openvino::runtime)

# Helper message
message(STATUS "Configured to build Intel NPU Library with OpenVINO")
