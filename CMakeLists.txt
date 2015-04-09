cmake_minimum_required(VERSION 2.8)

project(gpu-bake)

if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else ()
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	#set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif ()

# Setup externals

set(EIGEN_INCLUDE_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
include_directories(${EIGEN_INCLUDE_DIR})

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIR})
link_libraries(${OpenCL_LIBRARY})

find_package(OpenSceneGraph REQUIRED osgDB osgViewer osgGA osgUtil)
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
link_libraries(${OPENSCENEGRAPH_LIBRARIES})

find_package(OpenCV)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_libraries(${OpenCV_LIBS})
    add_definitions(-DBAKE_WITH_OPENCV)
endif ()

# Setup library

set(GPUBAKE_FILES 
	inc/bake/log.h 
	inc/bake/stringify.h
	inc/bake/surface.h
	inc/bake/image.h
	inc/bake/convert_surface.h
	src/convert_surface.cpp	
)

set(GPUBAKE_OPENCL_FILES
	inc/bake/opencl/bake.h
	inc/bake/opencl/bake.cl
	src/opencl/bake.cpp
)

source_group(bake FILES ${GPUBAKE_FILES})
source_group(bake\\opencl FILES ${GPUBAKE_OPENCL_FILES})
	
include_directories(inc)
add_library(gpubake ${GPUBAKE_FILES} ${GPUBAKE_OPENCL_FILES})

# Setup examples

set(GPUBAKE_EXAMPLE_FILES
    examples/main.cpp
	examples/example_bake_colors.cpp
	examples/example_osg_shaders.cpp
)

include_directories(examples)
add_executable(gpubake_examples ${GPUBAKE_EXAMPLE_FILES})
target_link_libraries(gpubake_examples gpubake)