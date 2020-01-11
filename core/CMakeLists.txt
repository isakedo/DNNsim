cmake_minimum_required(VERSION 3.1)
project(
        core
        VERSION 1.0.0
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/core/Dataflow.h
        include/core/Architecture.h
        src/Utils.cpp
        src/OutputStationary.cpp
        src/WindowFirstOutS.cpp
        src/Simulator.cpp
        src/DaDianNao.cpp
        src/Stripes.cpp
        src/ShapeShifter.cpp
        src/Loom.cpp
        src/BitPragmatic.cpp
        src/Laconic.cpp
        src/BitTactical.cpp
        src/SCNN.cpp
)


target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE base
        PRIVATE interface
        PRIVATE sys
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)