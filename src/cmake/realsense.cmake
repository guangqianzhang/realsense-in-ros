find_package(realsense2 2.50.0)
if(NOT realsense2_FOUND)
    message(FATAL_ERROR "\n\n Intel RealSense SDK 2.0 is missing, please install it from https://github.com/IntelRealSense/librealsense/releases\n\n")
endif()

set(DEPENDENCIES  glfw realsense2)
include_directories(../third-party/tclap/include  ${realsense2_INCLUDE_DIR})
list(APPEND ALL_TARGET_LIBRARIES ${realsense2_LIBRARY})