set(CMAKE_CXX_STANDARD 17)
find_package(TensorflowCC REQUIRED)

find_package(/usr/local//lib/python3.11/site-packages/tensorflow/include/tensorflow/core/protobuf)

#include_directories(/usr/local/bin/protoc-3/include)
#include_directories(/usr/local/Cellar/protobuf@3/3.20.3/include)
include_directories(/usr/local/include)
add_executable(example example.cpp)

# Link the Tensorflow library.
target_link_libraries(example TensorflowCC::TensorflowCC)
#target_link_libraries(example /usr/local/lib/protoc-3/lib/libprotobuf.dylib)
#target_link_libraries(example /usr/local/Cellar/protobuf@3/3.20.3/lib/libprotobuf.dylib)

target_link_libraries(example /usr/local/lib/libprotobuf.dylib)
# You may also link cuda if it is available.
# find_package(CUDA)
# if(CUDA_FOUND)
#   target_link_libraries(example ${CUDA_LIBRARIES})
# endif()
