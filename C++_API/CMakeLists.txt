set(CMAKE_CXX_STANDARD 17)
find_package(TensorflowCC REQUIRED)
# change this to where protobuf is located on your computer
find_package(/usr/local//lib/python3.11/site-packages/tensorflow/include/tensorflow/core/protobuf)

include_directories(/usr/local/include)
add_executable(example example.cpp)

# Link the Tensorflow library.
target_link_libraries(example TensorflowCC::TensorflowCC)

target_link_libraries(example /usr/local/lib/libprotobuf.dylib)
# You may also link cuda if it is available.
# find_package(CUDA)
# if(CUDA_FOUND)
#   target_link_libraries(example ${CUDA_LIBRARIES})
# endif()
