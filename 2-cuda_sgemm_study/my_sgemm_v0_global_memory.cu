#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
void rand_matrix(float *matrix, int row, int col) {
   

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // 将 2D 坐标 (i, j) 映射到 1D 线性空间
            matrix[i * col + j] = (float)rand() / (float)RAND_MAX;
        }
    }
}
void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K){

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = sum;
        }
    }
}
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K){

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < N && y < M) {
        // 获取block开始位置
        // blockDim.y * blockIdx.y代表需要跳多少行
        // K 每行有几个元素
        float *A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;
        // blockDim.x * blockIdx.x 代表 需要跳多少列
        float *B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

        float temp = 0.0f;
        for (int k = 0; k < K; k++) {
            temp += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
        }
        // 写回
        C_ptr[y * N + x] = temp;
    }

}
float compare_matrics(float *A, float *B, int M, int N){

    float max_err = 0.0f;

    for (int m = 0; m < M; m++) {

        for (int n = 0; n < N; n++) {

            float err = fabs(A[m * N + n] - B[m * N + n]);
            if(err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}
int main()
{

    printf("my_sgemm_v0_global_memory\n");
     // 设置随机数种子，确保每次运行生成的结果不同
    // 如果调试时想要结果可复现，可以把这行注掉
    srand(time(NULL)); 
    
    int m = 512;
    int n = 512;
    int k = 512;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    // cpu相关变量
    float *matrix_host_A = (float *)malloc(mem_size_A);
    float *matrix_host_B = (float *)malloc(mem_size_B);

    float *matrix_cpu_cacl_C = (float *)malloc(mem_size_C);
    float *matrix_gpu_cacl_C = (float *)malloc(mem_size_C);
    // 初始化
    rand_matrix(matrix_host_A, m, k);
    rand_matrix(matrix_host_B, k, n);
    memset(matrix_cpu_cacl_C, 0, mem_size_C);
    memset(matrix_gpu_cacl_C, 0, mem_size_C);
    // cpu 进行 sgemm
    cpu_sgemm(matrix_host_A, matrix_host_B, matrix_cpu_cacl_C, m, n, k);

    // gpu相关变量
    float *matrix_device_A, *matrix_device_B, *matrix_device_C;
    cudaMalloc((void **)&matrix_device_A, mem_size_A);
    cudaMalloc((void **)&matrix_device_B, mem_size_B);
    cudaMalloc((void **)&matrix_device_C, mem_size_C);
    // 初始化
    cudaMemcpy(matrix_device_A, matrix_host_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_device_B, matrix_host_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_device_C, matrix_gpu_cacl_C, mem_size_C, cudaMemcpyHostToDevice);

    // gpu计算
    constexpr int BLOCK = 8;
    dim3 block(BLOCK, BLOCK);
    // 把代表列的 n 放在第一个参数(x)，代表行的 m 放在第二个参数(y)
    dim3 grid((n - 1 + BLOCK) / BLOCK, (m - 1 + BLOCK) / BLOCK);
    cuda_sgemm<<<grid, block>>>(matrix_device_A, matrix_device_B, matrix_device_C, m, n, k);

    // 进行比较
    cudaMemcpy(matrix_gpu_cacl_C, matrix_device_C, mem_size_C,cudaMemcpyDeviceToHost);
    float diff = compare_matrics(matrix_cpu_cacl_C, matrix_gpu_cacl_C, m, n);
    if(diff < 1e-4) {
        printf("Result Check: PASS! Max Error = %f\n", diff);
    } else {
        printf("Result Check: FAILED! Max Error = %f\n", diff);
    }
    // 释放相关内存
    free(matrix_host_A);
    free(matrix_host_B);
    free(matrix_cpu_cacl_C);
    free(matrix_gpu_cacl_C);
    cudaFree(matrix_device_A);
    cudaFree(matrix_device_B);
    cudaFree(matrix_device_C);
    return 0;
}