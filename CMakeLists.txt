
cmake_minimum_required(VERSION 2.8)
project( extractHog )
find_package( OpenCV REQUIRED )
#find_package( CUDA REQUIRED )

IF (CUDA_FOUND)
	add_compile_options(-DHAVE_CUDA)
ENDIF()

add_compile_options(-g -std=c++11)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MY_COMPILE_FLAGS}" )

add_executable( extractHog main.cpp )
target_link_libraries( extractHog ${OpenCV_LIBS} tbb)

