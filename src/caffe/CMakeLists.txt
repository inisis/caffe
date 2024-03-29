# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

# add_library(caffe ${srcs})
# target_link_libraries(caffe proto -lglog -lprotobuf -lboost_thread -lboost_system -lboost_filesystem -lboost_regex -lhdf5_hl -lhdf5 -lopenblas)
# caffe_default_properties(caffe)
# target_include_directories(caffe PUBLIC
#                                  $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
#                                  $<INSTALL_INTERFACE:include>)
# set_target_properties(caffe PROPERTIES
#     VERSION   ${CAFFE_TARGET_VERSION}
#     SOVERSION ${CAFFE_TARGET_SOVERSION}
#     )

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${proto_hdrs} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe/proto)
# install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)
