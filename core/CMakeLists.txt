cmake_minimum_required(VERSION 3.1)
project(
        core
        VERSION 1.0.0
        LANGUAGES CXX
)

add_custom_target(
        libdramsim_make
        COMMAND make libdramsim.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../DRAMSim2
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/core/Memory.h
        include/core/Architecture.h
        src/Utils.cpp
        src/Control.cpp
        src/OutputStationary.cpp
        src/WindowFirstOutS.cpp
        src/DRAM.cpp
        src/GlobalBuffer.cpp
        src/DaDianNao.cpp
        src/Stripes.cpp
        src/ShapeShifter.cpp
        src/Loom.cpp
        src/BitPragmatic.cpp
        src/Laconic.cpp
        src/BitTactical.cpp
        src/SCNN.cpp
        src/Composer.cpp
        src/PPU.cpp
        src/Simulator.cpp
)

add_dependencies(
        ${PROJECT_NAME}
        libdramsim_make
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC include/
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../DRAMSim2/
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE base
        PRIVATE sys
        ${CMAKE_CURRENT_SOURCE_DIR}/../DRAMSim2/libdramsim.so
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)