cmake_minimum_required(VERSION 3.20.0)

# 1. 必须在 project() 之前设置编译器路径
set(CMAKE_CUDA_COMPILER "/home/yechangxin/cuda-12.4/bin/nvcc")

# --- 关键修改：为 RTX 5880 指定 Ada Lovelace 架构 ---
set(CMAKE_CUDA_ARCHITECTURES "89")

project(cuda_pratice VERSION 0.1.0 LANGUAGES CUDA CXX C)



# 2. 设置 CUDAToolkit 的查找根目录
set(CUDAToolkit_ROOT "/home/yechangxin/cuda-12.4")

find_package(CUDAToolkit REQUIRED) # 建议加上 REQUIRED，找不到时直接报错

# add_subdirectory(1-cuda_reduce_study)
add_subdirectory(2-cuda_sgemm_study)