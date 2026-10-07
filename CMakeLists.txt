cmake_minimum_required(VERSION 3.15)

project(statcore VERSION 0.1
                DESCRIPTION "Some statistical tooling."
                LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)      # build/compile_commands.json for VS Code

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Armadillo REQUIRED)

# # header only library
# add_library(agg INTERFACE)
# target_include_directories(agg INTERFACE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(agg INTERFACE ${ARMADILLO_LIBRARIES})

# compiled library
add_library(agg src/agg.cpp)
target_include_directories(agg PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(agg PUBLIC ${ARMADILLO_LIBRARIES})

# create scratch executable
add_executable(scratch scratch/scratch.cpp)
target_link_libraries(scratch PRIVATE agg)
target_compile_options(scratch PRIVATE -Wall -Wextra -Wpedantic)

# add_library(statcore STATIC src/main.cpp)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Armadillo REQUIRED)

# add_executable(app main.cpp)
# target_include_directories(app PRIVATE ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(app PRIVATE ${ARMADILLO_LIBRARIES})