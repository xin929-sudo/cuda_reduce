# 学习cuda编程
## 1-cuda_reduce_study
## 2-cuda_sgemm_study
### my_sgemm_v1_shared_memory
在 CUDA 中，__syncthreads() 的严格定义是：同一个 Block 内的所有线程，必须全部到达这个同步点，才能继续往下执行。
假设你的矩阵是 $100 \times 100$，而你的 Block 大小是 $32 \times 32$。当处理到矩阵边缘的 Block 时，有些线程的坐标可能超出了矩阵（比如 $x=120, y=120$）。那些超出边缘的线程，因为 if(x < N && y < M) 为假，它们直接跳过了整个 if 块，执行结束了。而那些在矩阵内部的线程，进入了 if 块，并在 __syncthreads() 这里停下来，苦苦等待 Block 里的其他人。
结果就是：等的人永远等不到，走的人也不会再回来。整个 Block 永久卡死，程序崩溃！
### my_sgemm_v3_using_float4
在 CUDA中，内存合并访问，只要一个warp中存在某几个线程读取内存的时候总和为128KB的时候，就可以合并内存访问
flaot4 就是 可以一次性读取4个float的数据