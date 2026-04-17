#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 设为 1 开启高性能外积法，设为 0 退回内积法
#define USE_OUTER_PRODUCT 1

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
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int M_NUM_PER_THREAD,
         unsigned int N_NUM_PER_THREAD,
         unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 获取block开始位置
    float *A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[2][K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float b_shared[2][K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    register float reg_a[M_NUM_PER_THREAD] = {0.0f};
    register float reg_b[N_NUM_PER_THREAD] = {0.0f};
    // 辅助读取
    register float a_load_reg[K_NUM_PER_THREAD] = {0.0f};
    // 每个线程需要计算
    register float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.0f};
   
    // 定义读写指针
    int write_idx = 0;
    int read_idx = 0;
    // 序幕--预读取 0 块 数据
    // 搬运 A 矩阵
    for (int m = 0; m < M_NUM_PER_THREAD; m++) {

        FETCH_FLOAT4(a_load_reg[0]) = FETCH_FLOAT4(A_ptr_start[(ty * M_NUM_PER_THREAD + m) * K + tx * K_NUM_PER_THREAD]);

        #pragma unroll
        for (int k = 0; k < K_NUM_PER_THREAD; k++) {
            a_shared[write_idx][tx * K_NUM_PER_THREAD + k][ty * M_NUM_PER_THREAD + m] = a_load_reg[k];
        }
    }
    // 搬运 B 矩阵
    for (int k = 0; k < K_NUM_PER_THREAD; k++) {

        FETCH_FLOAT4(b_shared[write_idx][ty * K_NUM_PER_THREAD + k][tx * N_NUM_PER_THREAD]) =
            FETCH_FLOAT4(B_ptr_start[(ty * K_NUM_PER_THREAD + k) * N + tx * N_NUM_PER_THREAD]);
    }
    // 线程同步,等待 第一块数据读取完毕
    __syncthreads();

    // 主循环 Ping-Pong 并发掩盖延迟
    for (int s = K_NUM_PER_BLOCK; s < K; s += K_NUM_PER_BLOCK) {

        // 切换写指针
        write_idx = 1 - write_idx;

        // 动作 A：搬运工去 Global 拿下一块数据 (步长为 s) 到 write_idx
        // 搬运 A 矩阵
        for (int m = 0; m < M_NUM_PER_THREAD; m++) {
            
            FETCH_FLOAT4(a_load_reg[0]) = FETCH_FLOAT4(A_ptr_start[(ty * M_NUM_PER_THREAD + m) * K + tx * K_NUM_PER_THREAD + s]);
            
            #pragma unroll
            for (int k = 0; k < K_NUM_PER_THREAD; k++) {
                a_shared[write_idx][tx * K_NUM_PER_THREAD + k][ty * M_NUM_PER_THREAD + m] = a_load_reg[k];
            }
        }
        // 搬运 B 矩阵
        for (int k = 0; k < K_NUM_PER_THREAD; k++) {
            FETCH_FLOAT4(b_shared[write_idx][ty * K_NUM_PER_THREAD + k][tx * N_NUM_PER_THREAD]) =
                FETCH_FLOAT4(B_ptr_start[(ty * K_NUM_PER_THREAD + k + s) * N + tx * N_NUM_PER_THREAD]);
        }
        // 不要线程同步
        // __syncthreads();


        // 👉 动作 B：计算工在 read_idx 上做外积计算！
        // 🔥 注意：动作 A 和 动作 B 之间绝对不能有 __syncthreads()！让它们并发！
        for (int k = 0; k < K_NUM_PER_BLOCK; ++k) {

            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[read_idx][k][ty * M_NUM_PER_THREAD]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[read_idx][k][tx * N_NUM_PER_THREAD]);
            // 开始计算
            #pragma unroll
            for (int m = 0; m < M_NUM_PER_THREAD; m++) {
                for (int n = 0; n < N_NUM_PER_THREAD; n++) {
                    temp[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // 所有人干完活，等待同步，准备进入下一轮
        __syncthreads();
        // 当前块算完了，把读指针也切到下一块
        read_idx = 1 - read_idx;
    }

    // 尾声：计算最后一块数据
    for (int k = 0; k < K_NUM_PER_BLOCK; ++k) {

        FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[read_idx][k][ty * M_NUM_PER_THREAD]);
        FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[read_idx][k][tx * N_NUM_PER_THREAD]);
        // 开始计算
        #pragma unroll
        for (int m = 0; m < M_NUM_PER_THREAD; m++) {
            for (int n = 0; n < N_NUM_PER_THREAD; n++) {
                temp[m][n] += reg_a[m] * reg_b[n];
            }
        }
    }
    // 结果写回去
    float *C_ptr_start = C_ptr + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for (int m = 0; m < M_NUM_PER_THREAD; ++m) {
        FETCH_FLOAT4(C_ptr_start[(ty * M_NUM_PER_THREAD + m) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[m][0]);
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

    printf("my_sgemm_v7_double_buffer.cu\n");
     // 设置随机数种子，确保每次运行生成的结果不同
    // 如果调试时想要结果可复现，可以把这行注掉
    // srand(time(NULL)); 
    
    int m = 1024;
    int n = 1024;
    int k = 1024;
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
    constexpr int M_NUM_PER_BLOCK = 64;
    constexpr int N_NUM_PER_BLOCK = 64;
    constexpr int K_NUM_PER_BLOCK = 64;

    constexpr int M_NUM_PER_THREAD = 4;
    constexpr int N_NUM_PER_THREAD = 4;
    constexpr int K_NUM_PER_THREAD = 4;

    dim3 block(16, 16);
    // 把代表列的 n 放在第一个参数(x)，代表行的 m 放在第二个参数(y)
    dim3 grid((n - 1 + N_NUM_PER_BLOCK) / N_NUM_PER_BLOCK, (m - 1 + M_NUM_PER_BLOCK) / M_NUM_PER_BLOCK);

    cuda_sgemm<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,M_NUM_PER_THREAD,N_NUM_PER_THREAD,K_NUM_PER_THREAD><<<grid, block>>>(matrix_device_A, matrix_device_B, matrix_device_C, m, n, k);

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