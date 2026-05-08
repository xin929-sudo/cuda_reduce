add_executable(combined_access combined_access.cu)
target_link_libraries(combined_access PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(combined_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

target_compile_options(combined_access PRIVATE -lineinfo)

# =====================================================================
# 🚀 自动化调优脚本配置：专属的 Nsight Compute 抓包目标
# =====================================================================
# add_custom_target(profile_combined_access
#     # 1. COMMAND: 用 bash 执行当前源码目录下的 run_ncu.sh，并把名字传给它
#     COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_ncu.sh combined_access
    
#     # 2. WORKING_DIRECTORY (极其关键): 
#     # 强制让 CMake 退回到源码目录去执行这个脚本。
#     # 因为我们的脚本里写了 "../build/..." 这种相对路径，必须在源码目录下执行才找得到！
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    
#     # 3. DEPENDS: 依赖关系保持不变。修改代码后跑脚本，会自动触发重新编译。
#     DEPENDS combined_access
    
#     # 4. COMMENT: 终端提示语
#     COMMENT "🚀 [工程化调优] 正在触发自定义脚本 run_ncu.sh 进行全自动抓包..."
# )

add_executable(my_transpose_v1_naive my_transpose_v1_naive.cu)
target_link_libraries(my_transpose_v1_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_transpose_v1_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

target_compile_options(my_transpose_v1_naive PRIVATE -lineinfo)