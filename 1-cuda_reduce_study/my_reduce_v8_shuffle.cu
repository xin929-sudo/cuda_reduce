#include<cstdio>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256


__device__ void warpReduce(volatile float* v_shared, unsigned int tid) {
     // 第一步： 32 折叠 为 16
        v_shared[tid] += v_shared[tid + 32];
        __syncwarp(); // 32 对齐进度

        // 第二步： 16 折叠 为 8
        v_shared[tid] += v_shared[tid + 16];
        __syncwarp();

        // 第三步： 8 折叠为 4
        v_shared[tid] += v_shared[tid + 8];
        __syncwarp();

        // 第四步： 4 折叠为 2
        v_shared[tid] += v_shared[tid + 4];
        __syncwarp();

        // 第五步： 2 折叠为 1
        v_shared[tid] += v_shared[tid + 2];
        __syncwarp();

        // 第六步： 总和
        v_shared[tid] += v_shared[tid + 1];
        __syncwarp();    
}


template <unsigned int NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output) {
    // 先把数据读到 共享内存里面，这样一个block里面都是共享的
    int tid = threadIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];
    // 1.获取每个block要处理数据的起始数组
    // 解决 idle 线程，线程一个线程需要处理原来的 NUM_PER_BLOCK倍
    float *input_start = d_input + blockIdx.x  * NUM_PER_BLOCK;
    
    float my_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++) {
        my_sum += input_start[tid + THREAD_PER_BLOCK * i];
    }
    // 每32个线程进行直接寄存器操作，总共有 256 / 32 = 8 个 warp
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 16);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 8);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 4);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 2);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 1);

    __shared__ float warpLevelSums[32];
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    if(laneId == 0) { // 每个warp 的 0 号 线程 写入到 warpLevelSums 中
        warpLevelSums[warpId] = my_sum;
    }
    __syncthreads();
    // 第 0 组 warp 来 处理
    if(warpId == 0) {
        my_sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0;
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 16);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 8);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 4);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 2);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 1);
    }


    // 3. 一个线程写回
    if(tid == 0) {
        d_output[blockIdx.x] = my_sum;
    }
}

bool check(const float * a, const float *b, int N) {

    for(int i = 0; i < N; i++) {
        
        if(fabs(a[i] - b[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("my_reduce_v7_multi_add\n");
    const int N = 32 * 1024 * 1024;
    // cpu
    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_pre_thread = num_per_block / THREAD_PER_BLOCK;

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(block_num * sizeof(float));
    float *h_gpu_result = (float*)malloc(block_num * sizeof(float));
    // init data
    for(int i = 0; i < N; ++i) {
        h_input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for(int i = 0; i < block_num; i++) {
        float cur = 0.0f; // 当前block 的总和
        for(int j = 0; j < num_per_block; ++j) {
            cur += h_input[i * num_per_block + j];
        }
        h_result[i] = cur;
    }
    // gpu
    float *d_input, *d_result;
    cudaMalloc((void **)&d_input,N * sizeof(float));
    cudaMalloc((void **)&d_result,block_num * sizeof(float));

    // 从 cpu 拷贝到 gpu
    cudaMemcpy(d_input,h_input,N * sizeof(float),cudaMemcpyHostToDevice);

    // 配置 线程
    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);

    // for(int i = 0; i < 50; i++) {
    //     reduce<<<Grid, Block>>>(d_input,d_result);
    // }
    reduce<num_per_block,num_pre_thread><<<Grid, Block>>>(d_input,d_result);
    // 拷贝回cpu
    cudaMemcpy(h_gpu_result, d_result, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(h_result, h_gpu_result,block_num)){
        printf("The ans is right.\n");
    } else {
        printf("The ans is wrong.\n");
        for(int i = 0; i < block_num; i++) {
            printf("%lf ", h_result[i]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_result);
    free(h_input);
    free(h_gpu_result);
    free(h_result);

    return 0;
}