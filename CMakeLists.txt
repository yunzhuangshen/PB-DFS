cmake_minimum_required(VERSION 2.8.4)

project(CO)

set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} -Wno-terminate -std=c++11 -g -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/build")
FILE(GLOB SOURCE_FILES src/*.cpp)

find_package(SCIP REQUIRED)

add_executable( CO ${SOURCE_FILES})

target_include_directories(CO PRIVATE 
						"${SCIP_INCLUDE_DIRS}"
						"${CMAKE_CURRENT_SOURCE_DIR}/"
						)

target_link_libraries(CO ${SCIP_LIBRARIES})