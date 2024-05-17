set(CMAKE_CXX_STANDARD 17)
find_package(TensorflowCC REQUIRED)
find_package(/usr/local//lib/python3.11/site-packages/tensorflow/include/tensorflow/core/protobuf)
include_directories(/usr/local/include)

include_directories(DyMonD/include)
add_executable(example c++api_general.cpp)

target_link_libraries(example pcap /usr/local/opt/boost /usr/local/opt/libpthread-stubs)
target_link_libraries(example TensorflowCC::TensorflowCC)
target_link_libraries(example /usr/local/lib/libprotobuf.dylib)
