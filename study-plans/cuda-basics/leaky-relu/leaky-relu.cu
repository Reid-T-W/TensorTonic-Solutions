#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write code here
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        float x = input[thread_id];
        output[thread_id] = fmaxf(x, alpha * x);
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}