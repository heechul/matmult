set(PROGRAM_NAME matrix_mult)

add_subdirectory(eigen)

add_executable(${PROGRAM_NAME}
        main.cpp
        )
target_include_directories(${PROGRAM_NAME} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        )
        
target_link_libraries(${PROGRAM_NAME}
        pico_stdlib
        eigen
        )
pico_enable_stdio_usb(${PROGRAM_NAME} 1)
pico_enable_stdio_uart(${PROGRAM_NAME} 0)
pico_add_extra_outputs(${PROGRAM_NAME})
