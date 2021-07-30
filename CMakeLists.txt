# CMakeList.txt: OpencvAndHalcon 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project(OpencvAndHalcon)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置为release模式,在vs编译器编译的情况下,debug模式和release模式的性能存在较大的差异
set(CMAKE_BUILD_TYPE "Release")

# 开启此选项,可以在release模式下进行debug调试
if (WIN32)
    # 开启此选项,可以在release模式下进行debug调试
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
endif()

# 添加opencv支持
if (WIN32)
    set(OpenCV_DIR "C:/Program Files/CppLibrary/opencv/build")
endif()
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
#target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})  # 这个要放最后

# 添加OpenMP支持
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 添加halcon
include_directories("$ENV{HALCONROOT}/include"
                    "$ENV{HALCONROOT}/include/halconcpp")
link_directories("$ENV{HALCONROOT}/lib/$ENV{HALCONARCH}")

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME}   "OpencvAndHalcon.cpp"  "OpencvAndHalcon.h"  "timer.h" "thresholdTest.cpp" "thresholdTest.h" "hsvTest.cpp" "hsvTest.h" "halconConverter.h")

# TODO: 如有需要，请添加测试并安装目标。

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} halcon halconcpp)