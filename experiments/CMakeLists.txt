# Builds the three test copies of experiments/test_copies against an existing
# build of this polyDEAL source tree. Configure with
#
#   cmake -DDEAL_II_DIR=<deal.II installation> \
#         -DPOLYDEAL_BUILD=<build directory of this source tree> \
#         -DPARMETIS_DIR=<ParMETIS installation> \
#         <this directory>
#
# and build with make. Each test copy is one large translation unit; compile
# them one or two at a time on a machine with little memory.
cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.7.0 REQUIRED HINTS ${DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(polydeal_experiments CXX)

# The installed deal.II must provide these; nothing is downloaded or built here.
foreach(_feature MPI P4EST TRILINOS METIS)
  if(NOT DEAL_II_WITH_${_feature})
    message(FATAL_ERROR "deal.II at ${DEAL_II_PATH} was configured without "
                        "${_feature}, which the experiments need.")
  endif()
endforeach()

get_filename_component(POLYDEAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
set(POLYDEAL_BUILD "" CACHE PATH "Build directory of this polyDEAL source tree")
set(PARMETIS_DIR "" CACHE PATH "ParMETIS installation")
find_library(POLYDEAL_LIBRARY polydeal HINTS ${POLYDEAL_BUILD}/source
             REQUIRED NO_DEFAULT_PATH)
find_path(PARMETIS_INCLUDE_DIR parmetis.h HINTS ${PARMETIS_DIR}/include)
if(NOT PARMETIS_INCLUDE_DIR)
  message(FATAL_ERROR "ParMETIS not found. Pass -DPARMETIS_DIR=<installation>; "
                      "see the section ParMETIS of README.md.")
endif()
# ParMETIS ships its own METIS, which may differ from the one of deal.II. If the
# installation provides parmetis_private.o, an object that exports only the
# ParMETIS_* symbols, it is linked instead of the library to avoid a clash.
find_file(PARMETIS_PRIVATE_OBJECT parmetis_private.o
          HINTS ${PARMETIS_DIR}/lib64 ${PARMETIS_DIR}/lib NO_DEFAULT_PATH)
if(PARMETIS_PRIVATE_OBJECT)
  set(_parmetis ${PARMETIS_PRIVATE_OBJECT})
else()
  find_library(_parmetis parmetis HINTS ${PARMETIS_DIR}/lib64 ${PARMETIS_DIR}/lib
               REQUIRED)
endif()

foreach(_name repartitioned_poisson_test adaptive_heat_test
              adaptive_polytopal_timing)
  add_executable(${_name} test_copies/${_name}.cc)
  deal_ii_setup_target(${_name})
  target_include_directories(${_name} PRIVATE ${POLYDEAL_DIR}/include
                                              ${PARMETIS_INCLUDE_DIR})
  target_compile_definitions(${_name} PRIVATE
                             MESH_DIR="${POLYDEAL_DIR}/meshes")
  target_link_libraries(${_name} ${POLYDEAL_LIBRARY} ${_parmetis})
endforeach()
