cmake_minimum_required(VERSION 2.8)
project(line_detection6)
add_compile_options(-std=c++11)

set(OpenCV_DIR "usr/local/share/OpenCV/")
find_package(OpenCV 3.4 REQUIRED)
set(yaml-cpp_DIR "/usr/local/lib/cmake/yaml-cpp")
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${YAML_CPP_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "main.cpp" linedetector.cpp linesegment.cpp cameraprojection.cpp)

add_executable(projection_test projection_test.cpp cameraprojection.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES} ${Eigen3_LIBRARIES})
target_link_libraries(projection_test ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES} ${Eigen3_LIBRARIES})

