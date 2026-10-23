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
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(LAPACK REQUIRED)

# # header only library
# add_library(agg INTERFACE)
# target_include_directories(agg INTERFACE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(agg INTERFACE ${ARMADILLO_LIBRARIES})

##
# compiled libraries
##
add_library(agg src/agg.cpp)
target_include_directories(agg PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(agg PUBLIC ${ARMADILLO_LIBRARIES})

# -O3 -fopenmp-simd -mcpu=apple-m2
add_library(matmult src/matmult.cpp)
target_include_directories(matmult PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(matmult PUBLIC ${ARMADILLO_LIBRARIES})
target_compile_options(matmult PRIVATE -O3 -fopenmp-simd -mcpu=apple-m2)

add_library(models src/models.cpp)
target_include_directories(models PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(models PUBLIC ${ARMADILLO_LIBRARIES})

add_library(logger src/logger.cpp)
target_include_directories(
  logger PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ARMADILLO_INCLUDE_DIRS}
)
target_link_libraries(models PUBLIC ${ARMADILLO_LIBRARIES})

add_library(statcore INTERFACE)
target_link_libraries(statcore INTERFACE agg matmult models LAPACK::LAPACK BLAS::BLAS)
target_compile_options(statcore INTERFACE -g)
target_link_options(statcore INTERFACE)

##
# other executables
##

add_executable(scratch scratch/scratch.cpp)
target_link_libraries(scratch PRIVATE agg matmult)
target_link_libraries(scratch PRIVATE Eigen3::Eigen)
target_compile_options(scratch PRIVATE -Wall -Wextra -Wpedantic)

add_executable(sum_bmk bench/sum_bmk.cpp)
target_link_libraries(sum_bmk PRIVATE agg)
target_compile_options(sum_bmk PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mat_bmk bench/mat_bmk.cpp)
target_link_libraries(mat_bmk PRIVATE matmult)
target_compile_options(mat_bmk PRIVATE -Wall -Wextra -Wpedantic)

add_executable(model_tests tests/model_tests.cpp)
target_link_libraries(model_tests PRIVATE statcore)
target_link_libraries(model_tests PRIVATE models)
target_compile_options(model_tests PRIVATE -O3 -Wall -Wextra -Wpedantic)

# add_library(statcore STATIC src/main.cpp)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Armadillo REQUIRED)

# add_executable(app main.cpp)
# target_include_directories(app PRIVATE ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(app PRIVATE ${ARMADILLO_LIBRARIES})