cmake_minimum_required(VERSION 3.1)
project(
        sys
        VERSION 1.0.0
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/sys/cxxopts.h
        include/sys/common.h
        src/Stats.cpp
        src/Batch.cpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
        ${ProtobufIncludePath}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE proto
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)