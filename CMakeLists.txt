cmake_minimum_required(VERSION 3.12)
project(DNNsim)

set(CMAKE_CXX_STANDARD 17)

# Add library for load the inputs
#add_subdirectory(loader)
# Add the core library
#add_subdirectory(core)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(
        ${PROJECT_NAME}
        main.cpp
)