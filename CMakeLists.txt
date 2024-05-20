# CMake entry point
cmake_minimum_required (VERSION 3.0)

# We need CUDA.
project (RotationEstimationGPGPU LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# We need OpenGL.
find_package(OpenGL REQUIRED)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

# Compile external dependencies 
add_subdirectory (external)

# On Visual 2005 and above, this module can set the debug working directory
cmake_policy(SET CMP0026 OLD)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
include(CreateLaunchers)
include(MSVCMultipleProcessCompile) # /MP

if(INCLUDE_DISTRIB)
	add_subdirectory(distrib)
endif(INCLUDE_DISTRIB)

include_directories(
	external/AntTweakBar-1.16/include/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	.
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

add_executable(RotationEstimationGPGPU 
	src/Demo.cpp

	src/Tests.cpp
	src/Tests.h

	src/EstimateRotation.cu
	src/EstimateRotation.h

	src/DeviceBuffer.cu

	src/Utilities.cpp
	src/Utilities.h
)

set_target_properties(RotationEstimationGPGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(RotationEstimationGPGPU PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(RotationEstimationGPGPU
	${ALL_LIBS}
	ANTTWEAKBAR_116_OGLCORE_GLFW
)
