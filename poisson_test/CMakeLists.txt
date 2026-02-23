#cmake -Ddeal.II_DIR=/home/wzk/dealii/dealii/deal.II-v9.7.0/lib/cmake/deal.II .
cmake_minimum_required(VERSION 3.13)
project(poisson_minimal)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(deal.II REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCE_FILES
  test/poisson.cc
  source/agglomeration_handler.cc
  source/fe_agglodgp.cc
  source/mapping_box.cc
)

add_executable(poisson ${SOURCE_FILES})

target_compile_definitions(poisson PUBLIC BOOST_MATH_DISABLE_FLOAT128)

DEAL_II_SETUP_TARGET(poisson)