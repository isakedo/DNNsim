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
        include/core/Utils.h
        include/core/Architecture.h
        include/core/Memory.h
        include/core/Laconic.h
        include/core/BitPragmatic.h
        include/core/Stripes.h
        include/core/ShapeShifter.h
        include/core/Loom.h
        include/core/BitTactical.h
        include/core/BitTacticalP.h
        include/core/BitTacticalE.h
        include/core/TensorDash.h
        include/core/SCNN.h
        src/Utils.cpp
        src/Memory.cpp
        src/Simulator.cpp
        src/Laconic.cpp
        src/BitPragmatic.cpp
        src/Stripes.cpp
        src/BitTactical.cpp
        src/BitTacticalP.cpp
        src/BitTacticalE.cpp
        src/TensorDash.cpp
        src/ShapeShifter.cpp
        src/Loom.cpp
        src/SCNN.cpp
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
        PRIVATE interface
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