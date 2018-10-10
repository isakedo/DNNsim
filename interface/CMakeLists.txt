cmake_minimum_required(VERSION 3.12)
project(
        interface
        VERSION 0.0.1
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/interface/NetReader.h
        include/interface/NetWriter.h
        src/NetReader.cpp
        src/NetWriter.cpp)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE cnpy
        PRIVATE core
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)