# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Define the project name and version
project(MyProject VERSION 1.0 LANGUAGES CXX)

# Find the OpenCV package
# find_package(OpenCV REQUIRED)

# Include directories for OpenCV
# Setup the opencv
set(OpenCV_DIR /home/nhattt/anaconda3/envs/cpp/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(l0_norm)
add_subdirectory(python)

# Define the executable name and its source files
# add_executable(my_program l0_norm/main.cpp)

# Link the OpenCV libraries to the executable
# target_link_libraries(my_program ${OpenCV_LIBS})
# Link the OpenCV libraries to the executable
# target_link_libraries(my_program ${OpenCV_LIBS})


# Set C++ standard
# set_target_properties(my_program PROPERTIES
#     CXX_STANDARD 11
#     CXX_STANDARD_REQUIRED YES
# )
