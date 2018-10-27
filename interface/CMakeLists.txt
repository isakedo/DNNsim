cmake_minimum_required(VERSION 3.1)
project(
        interface
        VERSION 0.0.3
        LANGUAGES CXX
)

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(
        PROTO_GENERATED_SOURCES
        PROTO_GENERATED_HEADERS
        proto/network.proto
        proto/caffe.proto
)

# Create library
add_library(
        ${PROJECT_NAME}
        ${PROTO_GENERATED_SOURCES}
        ${PROTO_GENERATED_HEADERS}
        include/interface/NetReader.h
        include/interface/NetWriter.h
        src/NetReader.cpp
        src/NetWriter.cpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC include/
)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC ${Protobuf_LIBRARIES}
        PRIVATE cnpy
        PRIVATE core
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