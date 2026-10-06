cmake_minimum_required(VERSION 3.15)

project(statcore VERSION 0.1
                DESCRIPTION "Some statistical tooling."
                LANGUAGES CXX)

add_library(statcore STATIC src/main.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

add_executable(app main.cpp)
target_include_directories(app PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(app PRIVATE ${ARMADILLO_LIBRARIES})