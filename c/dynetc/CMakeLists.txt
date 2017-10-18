set(HEADER_FILES c_api.h c_api_internal.h)
set(SOURCE_FILES c_api.cc)
add_library(dynetc STATIC ${HEADER_FILES} ${SOURCE_FILES})
