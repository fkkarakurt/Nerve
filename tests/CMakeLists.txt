cmake_minimum_required(VERSION 3.14)

# --------------------------------------------------------------------------
# Tests link against libnerveNET (the compiled static library from src/).
# If NERVE_BUILD_STATIC is OFF the src/ subdirectory was not added, so we
# fall back to the header-only interface.
# --------------------------------------------------------------------------
if(TARGET nerve::nerveNET)
    set(NERVE_TEST_LIB nerve::nerveNET)
else()
    # Header-only fallback: each TU must define NERVE_IMPLEMENTATION
    set(NERVE_TEST_LIB nerve::nerve)
endif()

function(add_nerve_test target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(${target} PRIVATE ${NERVE_TEST_LIB})
endfunction()

add_nerve_test(benchmark      benchmark.c)
add_nerve_test(create_network create_network.c)
add_nerve_test(copy_network   copy_network.c)
add_nerve_test(show_network   show_network.c spec.c)
add_nerve_test(train_network  train_network.c spec.c)
add_nerve_test(jolt_network   jolt_network.c)

# --------------------------------------------------------------------------
# CTest integration
# --------------------------------------------------------------------------
enable_testing()
add_test(NAME xor_convergence
    COMMAND train_network -n3 -e0.0001 ${CMAKE_CURRENT_SOURCE_DIR}/xornet.spec
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
