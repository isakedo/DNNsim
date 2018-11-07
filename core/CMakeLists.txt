cmake_minimum_required(VERSION 3.1)
project(
        core
        VERSION 0.0.3
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/core/Layer.h
        include/core/Network.h
        include/core/Simulator.h
        include/core/InferenceSimulator.h
        include/core/Laconic.h
        include/core/BitPragmatic.h
        include/core/BitFusion.h
        src/Simulator.cpp
        src/InferenceSimulator.cpp
        src/Laconic.cpp
        src/BitPragmatic.cpp
        src/BitFusion.cpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
)


target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE cnpy
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