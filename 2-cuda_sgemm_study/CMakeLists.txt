
# 添加可执行文件
add_executable(my_sgemm_v0_global_memory my_sgemm_v0_global_memory.cu)

# 链接库
target_link_libraries(my_sgemm_v0_global_memory PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()


# 添加可执行文件
add_executable(my_sgemm_v1_shared_memory my_sgemm_v1_shared_memory.cu)

# 链接库
target_link_libraries(my_sgemm_v1_shared_memory PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v2_increase_work_of_per_thread   my_sgemm_v2_increase_work_of_per_thread.cu)

# 链接库
target_link_libraries(my_sgemm_v2_increase_work_of_per_thread   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v2_increase_work_of_per_thread   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v3_using_float4   my_sgemm_v3_using_float4.cu)

# 链接库
target_link_libraries(my_sgemm_v3_using_float4   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v3_using_float4   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v4_using_float4_and_register   my_sgemm_v4_using_float4_and_register.cu)

# 链接库
target_link_libraries(my_sgemm_v4_using_float4_and_register   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v4_using_float4_and_register   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v5_using_register_2d   my_sgemm_v5_using_register_2d.cu)

# 链接库
target_link_libraries(my_sgemm_v5_using_register_2d   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v5_using_register_2d   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v6_smem_transpose   my_sgemm_v6_smem_transpose.cu)

# 链接库
target_link_libraries(my_sgemm_v6_smem_transpose   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v6_smem_transpose   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()

# 添加可执行文件
add_executable(my_sgemm_v7_double_buffer   my_sgemm_v7_double_buffer.cu)

# 链接库
target_link_libraries(my_sgemm_v7_double_buffer   PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v7_double_buffer   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)

endif()