# Build both Release and Debug versions of library and executables
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

file(GLOB APP_SOURCES ./*.cc )

# adaptive_polytopal reads adaptive_polytopal.prm from the working directory by
# default, so the file is copied next to the executables.
configure_file(adaptive_polytopal.prm
               ${CMAKE_CURRENT_BINARY_DIR}/adaptive_polytopal.prm COPYONLY)

# ParMETIS, for the examples that partition the polytope graph
# (include/polytope_repartition.h). Pass -DPARMETIS_DIR=/path/to/parmetis, or
# leave it empty to search the system paths; without ParMETIS these examples
# are skipped.
#
# ParMETIS ships its own METIS, which may differ from the one deal.II uses, and
# linking both can clash. If the installation provides parmetis_private.o, a
# relocatable object that exports only the ParMETIS_* symbols, it is linked
# instead of the libraries to avoid that. It is optional.
set(PARMETIS_DIR "" CACHE PATH "Installation directory of ParMETIS")
set(_parmetis_examples repartitioned_poisson adaptive_polytopal)
find_path(PARMETIS_INCLUDE_DIR parmetis.h HINTS ${PARMETIS_DIR}/include)
find_file(PARMETIS_PRIVATE_OBJECT parmetis_private.o
          HINTS ${PARMETIS_DIR}/lib64 ${PARMETIS_DIR}/lib NO_DEFAULT_PATH)
find_library(PARMETIS_LIBRARY parmetis
             HINTS ${PARMETIS_DIR}/lib64 ${PARMETIS_DIR}/lib)
find_library(PARMETIS_METIS_LIBRARY metis
             HINTS ${PARMETIS_DIR}/lib64 ${PARMETIS_DIR}/lib)
if(PARMETIS_INCLUDE_DIR AND (PARMETIS_PRIVATE_OBJECT OR PARMETIS_LIBRARY))
  set(_with_parmetis TRUE)
else()
  set(_with_parmetis FALSE)
  message(STATUS "ParMETIS not found: skipping ${_parmetis_examples}")
endif()

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")
      
      # Pick up the right library
      set(_lib "polydeal${_p}")

      foreach (testsourcefile ${APP_SOURCES})
      # Cut off the file extension and directory path
            get_filename_component( exename ${testsourcefile} NAME_WE )            
            
            # Require MUMPS for monodomain examples
            if(exename STREQUAL "monodomain_DG3D" OR exename STREQUAL "monodomain_DG2D")
              if(NOT DEAL_II_WITH_MUMPS)
                message(WARNING "\n"
                  "*** The example '${exename}' requires MUMPS support. ***\n\n"
                  "Please recompile deal.II with MUMPS enabled for this example to work correctly.\n"
                  "To enable MUMPS in deal.II, use -DDEAL_II_WITH_MUMPS=ON when configuring deal.II."
                )
              else()
                message("-- MUMPS support detected for ${exename}")
              endif()
            endif()
            
            list(FIND _parmetis_examples ${exename} _needs_parmetis)
            if(_needs_parmetis GREATER -1 AND NOT _with_parmetis)
              continue()
            endif()

            # Add suffix for debug
            set(testname ${exename}${_p})
            message("-- Configuring executable ${testname} (${testsourcefile})")

            add_executable( ${testname} ${testsourcefile} )

            # Make sure the library is linked to each app
            target_link_libraries(${testname} ${_lib})
            
            # Define MESH_DIR for the executable
            target_compile_definitions(${testname} PRIVATE MESH_DIR="${MESH_DIR}")
            
            DEAL_II_SETUP_TARGET(${testname} ${_BUILD_TYPE})

            if(_needs_parmetis GREATER -1)
              # After the system headers, so that <metis.h> stays the one
              # deal.II uses.
              target_compile_options(${testname} PRIVATE
                                     -idirafter ${PARMETIS_INCLUDE_DIR})
              if(PARMETIS_PRIVATE_OBJECT)
                target_link_libraries(${testname} ${PARMETIS_PRIVATE_OBJECT})
              else()
                target_link_libraries(${testname} ${PARMETIS_LIBRARY}
                                      ${PARMETIS_METIS_LIBRARY})
              endif()
            endif()
            INSTALL(TARGETS ${testname})
            set(testname ${testname}${_p})
            endforeach( testsourcefile ${APP_SOURCES} )
        endif ()
endforeach()
