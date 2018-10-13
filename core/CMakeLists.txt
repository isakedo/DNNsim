cmake_minimum_required(VERSION 3.1)
project(
        core
        VERSION 0.0.1
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/core/Layer.h
        include/core/Network.h
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
)


target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE cnpy
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)