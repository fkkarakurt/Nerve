cmake_minimum_required(VERSION 3.14)

# --------------------------------------------------------------------------
# Helper: each example is a single .c file that defines NERVE_IMPLEMENTATION
# and includes nerve.h directly, so the only dependency is nerve::nerve.
# --------------------------------------------------------------------------
function(add_nerve_example target src)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE nerve::nerve)
endfunction()

# --------------------------------------------------------------------------
# Core examples
# --------------------------------------------------------------------------
if(NERVE_BUILD_EXAMPLES)
    add_nerve_example(xor         01_xor.c)
    add_nerve_example(sine        02_sine.c)
    add_nerve_example(iris        03_iris.c)
    add_nerve_example(regression  05_regression.c)
    add_nerve_example(dropout     06_dropout.c)
    add_nerve_example(spiral      07_spiral.c)
    add_nerve_example(model_io    08_model_io.c)
    add_nerve_example(maintenance 09_predictive_maintenance.c)
endif()

# --------------------------------------------------------------------------
# Terminal AI games (neuroevolution, real-time ASCII display)
# --------------------------------------------------------------------------
if(NERVE_BUILD_GAMES)
    add_nerve_example(snake_ai  10_snake_ai.c)
    add_nerve_example(pong_ai   11_pong_ai.c)
    add_nerve_example(flappy_ai 12_flappy_ai.c)
endif()
