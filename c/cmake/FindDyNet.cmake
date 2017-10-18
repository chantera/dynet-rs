# - Try to find DyNet lib
#
# Once done this will define
#
#  DYNET_FOUND - system has dynet lib with correct version
#  DYNET_INCLUDE_DIR - the dynet include directory
#  DYNET_LIBRARY_DIR - the dynet library directory
#
# This module reads hints about search locations from
# the following enviroment variables:
#
# DYNET_ROOT
# DYNET_ROOT_DIR
#
if (DYNET_INCLUDE_DIR AND DYNET_LIBRARY_DIR)
  # in cache already
  set(DYNET_FOUND TRUE)

else ()

  # search first if an DyNetConfig.cmake is available in the system,
  # if successful this would set DYNET_INCLUDE_DIR and DYNET_LIBRARY_DIR
  # and the rest of the script will work as usual
  find_package(DyNet NO_MODULE QUIET)

  if(NOT DYNET_INCLUDE_DIR)
    find_path(DYNET_INCLUDE_DIR
      NAMES dynet/dynet.h
      HINTS
        ENV DYNET_ROOT
        ENV DYNET_ROOT_DIR
      PATHS
        ${CMAKE_INSTALL_PREFIX}/include
      )
  endif(NOT DYNET_INCLUDE_DIR)

  if(NOT DYNET_LIBRARY_DIR)
    find_path(DYNET_LIBRARY_DIR
      NAMES libdynet.so libdynet.dylib
      HINTS
        $ENV{DYNET_ROOT}/build/dynet
        $ENV{DYNET_ROOT_DIR}/build/dynet
      PATHS
        ${CMAKE_INSTALL_PREFIX}/lib
      )
  endif(NOT DYNET_LIBRARY_DIR)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(DyNet DEFAULT_MSG DYNET_INCLUDE_DIR DYNET_LIBRARY_DIR)

  mark_as_advanced(DYNET_INCLUDE_DIR)
  mark_as_advanced(DYNET_LIBRARY_DIR)

endif()

