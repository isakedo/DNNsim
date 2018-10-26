cmake_minimum_required(VERSION 3.1)
project(
        DNNsim
        VERSION 0.0.3
        LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add the math library
add_subdirectory(cnpy)
# Add the core library
add_subdirectory(core)
# Add library for load the inputs and dump the networks
add_subdirectory(interface)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(
        ${PROJECT_NAME}
        main.cpp
        cxxopts.h)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE cnpy
        PRIVATE core
        PRIVATE interface
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)