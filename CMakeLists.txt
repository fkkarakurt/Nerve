cmake_minimum_required(VERSION 3.14)
project(nerve
    # Keep in step with NERVENET_VERSION in nerve.h.
    VERSION 2.1.0
    DESCRIPTION "Zero-dependency neural networks in a single ANSI C header"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(NERVE_BUILD_EXAMPLES "Build all examples"       ON)
option(NERVE_BUILD_GAMES    "Build terminal AI games"  ON)
option(NERVE_BUILD_TESTS    "Build the test suite"     ON)

# --------------------------------------------------------------------------
# Header-only interface target  (the canonical usage)
#
#   FetchContent integration in user projects:
#
#     include(FetchContent)
#     FetchContent_Declare(nerve
#         GIT_REPOSITORY https://github.com/fkkarakurt/nerve.git
#         GIT_TAG        main
#     )
#     FetchContent_MakeAvailable(nerve)
#
#     target_link_libraries(myapp PRIVATE nerve::nerve)
# --------------------------------------------------------------------------
add_library(nerve INTERFACE)
add_library(nerve::nerve ALIAS nerve)

target_include_directories(nerve INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

find_library(NERVE_MATH_LIB m)
if(NERVE_MATH_LIB)
    target_link_libraries(nerve INTERFACE ${NERVE_MATH_LIB})
endif()

# --------------------------------------------------------------------------
# Examples and games
# --------------------------------------------------------------------------
if(NERVE_BUILD_EXAMPLES OR NERVE_BUILD_GAMES)
    add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------
# Tests  (ctest --test-dir build)
# --------------------------------------------------------------------------
if(NERVE_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_executable(test_nerve tests/test_nerve.c)
    target_link_libraries(test_nerve PRIVATE nerve::nerve)
    add_test(NAME nerve_core COMMAND test_nerve)
endif()

# --------------------------------------------------------------------------
# Install rules
# --------------------------------------------------------------------------
include(GNUInstallDirs)

install(FILES nerve.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS nerve
    EXPORT nerveTargets
)

install(EXPORT nerveTargets
    NAMESPACE nerve::
    FILE      nerveConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nerve
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nerveConfigVersion.cmake"
    VERSION     ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/nerveConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nerve
)
