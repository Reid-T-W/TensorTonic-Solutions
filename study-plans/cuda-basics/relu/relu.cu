#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Write code here
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        if (input[thread_id] > 0) {
            output[thread_id] = input[thread_id];
        } else {
            output[thread_id] = 0;         
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}