
cmake_minimum_required(VERSION 2.8)
project( extractHog )
find_package( OpenCV REQUIRED )
find_package(HDF5)

IF (CUDA_FOUND)
	add_compile_options(-DHAVE_CUDA)
ENDIF()

IF (HDF5_FOUND)
	include_directories(${HDF5_INCLUDE_DIR})
	add_compile_options(-DHAVE_HDF5)
	message("HDF5_CXX_LIBRARIES ${HDF5_CXX_LIBRARIES}")
	message("HDF5_INCLUDE_DIR ${HDF5_INCLUDE_DIR}")
	message("HDF5_C_LIBRARIES ${HDF5_C_LIBRARIES}")
	message("HDF5_LIBRARIES ${HDF5_LIBRARIES}")
ENDIF()

add_compile_options(-g -std=c++11)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MY_COMPILE_FLAGS}" )

add_executable( extractHog main.cpp )
target_link_libraries( extractHog ${OpenCV_LIBS} ${HDF5_C_LIBRARIES} tbb)

