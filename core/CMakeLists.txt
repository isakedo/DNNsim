cmake_minimum_required(VERSION 3.1)
project(
        core
        VERSION 1.0.0
        LANGUAGES CXX
)

# Create library
add_library(
        ${PROJECT_NAME}
        include/core/Layer.h
        include/core/Network.h
        include/core/Inference.h
        include/core/Laconic.h
        include/core/BitPragmatic.h
        include/core/Stripes.h
        include/core/DynamicStripes.h
        include/core/Loom.h
        include/core/BitTactical.h
        include/core/BitTacticalP.h
        include/core/BitTacticalE.h
        include/core/SCNN.h
        include/core/SCNNp.h
        include/core/SCNNe.h
        include/core/BitFusion.h
        src/Simulator.cpp
        src/Inference.cpp
        src/Laconic.cpp
        src/BitPragmatic.cpp
        src/Stripes.cpp
        src/BitTactical.cpp
        src/BitTacticalP.cpp
        src/BitTacticalE.cpp
        src/DynamicStripes.cpp
        src/Loom.cpp
        src/SCNN.cpp
        src/SCNNp.cpp
        src/SCNNe.cpp
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