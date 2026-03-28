

add_executable(my_reduce_v1_shared_memory my_reduce_v1_shared_memory.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v1_shared_memory PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v0_global_memory my_reduce_v0_global_memory.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v0_global_memory PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v2_no_divergence_branch my_reduce_v2_no_divergence_branch.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v2_no_divergence_branch PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v2_no_divergence_branch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v3_no_bank_conflict my_reduce_v3_no_bank_conflict.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v3_no_bank_conflict PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v3_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v4_add_during_load_plan_a my_reduce_v4_add_during_load_plan_a.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v4_add_during_load_plan_a PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v4_add_during_load_plan_a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v4_add_during_load_plan_b my_reduce_v4_add_during_load_plan_b.cu)

# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v4_add_during_load_plan_b PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v4_add_during_load_plan_b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 添加可执行文件
add_executable(my_reduce_v5_unroll_last_warp my_reduce_v5_unroll_last_warp.cu)
# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v5_unroll_last_warp PRIVATE CUDA::cudart CUDA::cublas)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v5_unroll_last_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 添加可执行文件
add_executable(my_reduce_v6_completely_unroll my_reduce_v6_completely_unroll.cu)
# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v6_completely_unroll PRIVATE CUDA::cudart CUDA::cublas)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v6_completely_unroll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 添加可执行文件
add_executable(my_reduce_v7_multi_add my_reduce_v7_multi_add.cu)
# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v7_multi_add PRIVATE CUDA::cudart CUDA::cublas)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v7_multi_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 添加可执行文件
add_executable(my_reduce_v8_shuffle my_reduce_v8_shuffle.cu)
# 1. 修复了拼写错误和链接方式，直接使用 CUDAToolkit 提供的 CUDA::cublas
target_link_libraries(my_reduce_v8_shuffle PRIVATE CUDA::cudart CUDA::cublas)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 2. 修复了生成器表达式的拼写和符号错误
    target_compile_options(my_reduce_v8_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


