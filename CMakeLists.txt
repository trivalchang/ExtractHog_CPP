
cmake_minimum_required(VERSION 2.8)
project( extractHog )
find_package( OpenCV REQUIRED )
find_package(HDF5)
find_package( LibXml2 REQUIRED )

IF (CUDA_FOUND)
	add_compile_options(-DHAVE_CUDA)
ENDIF()

include_directories(${HDF5_INCLUDE_DIR})
add_compile_options(-DHAVE_HDF5)
link_directories(${HDF5_LIBRARY_DIRS})

include_directories(${LIBXML2_INCLUDE_DIR})

add_compile_options(-g -std=c++11)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MY_COMPILE_FLAGS}" )
set(LIBS ${LIBS} ${LIBXML2_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} tbb hdf5_cpp)

add_executable( extractHog main.cpp )
target_link_libraries( extractHog ${LIBS} )

