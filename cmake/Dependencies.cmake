# These lists are later turned into target properties on main caffe library target
set(Caffe_LINKER_LIBS "")
set(Caffe_INCLUDE_DIRS "")
set(Caffe_DEFINITIONS "")
set(Caffe_COMPILE_OPTIONS "")

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

if(BUILD_python)
  set(HAVE_PYTHON TRUE)
  if(BUILD_python_layer)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
    list(APPEND Caffe_LINKER_LIBS PRIVATE Python3::Module Python3::NumPy)
  endif()
endif()