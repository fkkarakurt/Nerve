cmake_minimum_required(VERSION 3.14)

# --------------------------------------------------------------------------
# Generate interpolation.c from the generate_sigma utility
# --------------------------------------------------------------------------
add_executable(generate_sigma generate_sigma.c)
find_library(_M m)
if(_M)
    target_link_libraries(generate_sigma PRIVATE ${_M})
endif()
set_target_properties(generate_sigma PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/interpolation.c
    COMMAND $<TARGET_FILE:generate_sigma> > ${CMAKE_CURRENT_BINARY_DIR}/interpolation.c
    DEPENDS generate_sigma
    COMMENT "Generating sigmoid lookup table"
    VERBATIM
)

# --------------------------------------------------------------------------
# Static library — network.c includes interpolation.c directly
# --------------------------------------------------------------------------
add_library(nerveNET STATIC network.c)

# Ensure interpolation.c is generated before network.c is compiled
set_source_files_properties(network.c PROPERTIES
    OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/interpolation.c
)

target_include_directories(nerveNET
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}   # finds generated interpolation.c
)

find_library(_M2 m)
if(_M2)
    target_link_libraries(nerveNET PUBLIC ${_M2})
endif()

add_library(nerve::nerveNET ALIAS nerveNET)
