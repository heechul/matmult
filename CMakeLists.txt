cmake_minimum_required(VERSION 3.13)


set(PICO_BOARD pico_w)

include(pico_sdk_import.cmake)

project(matrix_mult C CXX ASM)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(EIGEN_MAX_ALIGN_BYTES 0)
add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DEIGEN_DONT_ALIGN)
add_definitions(-DEIGEN_DONT_VECTORIZE)


add_compile_options(
        -Wall
        -Werror
        )


# initialize the Raspberry Pi Pico SDK
pico_sdk_init()


add_subdirectory(pico)
