# python/CMakeLists.txt
find_package(pybind11 REQUIRED)

set(OpenCV_DIR /home/nhattt/anaconda3/envs/cpp/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create a shared library (Python module)
pybind11_add_module(l0_module wrapper.cpp)

# Link against the static library
target_link_libraries(l0_module PRIVATE l0_lib ${OpenCV_LIBS})
