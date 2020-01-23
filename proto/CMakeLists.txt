cmake_minimum_required(VERSION 3.1)
project(
        proto
        VERSION 1.0.0
        LANGUAGES CXX
)

find_package(Protobuf REQUIRED)

set(ProtobufIncludePath ${CMAKE_CURRENT_BINARY_DIR}
        CACHE INTERNAL "Path to generated protobuf files.")

protobuf_generate_cpp(
        PROTO_GENERATED_SOURCES
        PROTO_GENERATED_HEADERS
        caffe.proto
        batch.proto
)

# Create library
add_library(
        ${PROJECT_NAME}
        ${PROTO_GENERATED_SOURCES}
        ${PROTO_GENERATED_HEADERS}
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC include/
)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC ${Protobuf_LIBRARIES}
)

set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        COMPILE_FLAGS "${WARNING_FLAGS}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        LINKER_LANGUAGE CXX
)