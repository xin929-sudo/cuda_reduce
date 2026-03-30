
# 添加可执行文件
add_executable(my_sgemm_v0_global_memory my_sgemm_v0_global_memory.cu)

# 链接库
target_link_libraries(my_sgemm_v0_global_memory PRIVATE CUDA::cudart CUDA::cublas)

# --- 终极修改：使用 89-real，彻底禁止生成 PTX，杜绝 JIT 编译 ---
# set_target_properties(my_sgemm_v0_global_memory PROPERTIES CUDA_ARCHITECTURES "89-real")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 保持这里的调试信息标志不变
    target_compile_options(my_sgemm_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
endif()