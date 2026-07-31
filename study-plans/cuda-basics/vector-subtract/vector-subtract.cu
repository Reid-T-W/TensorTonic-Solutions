#include <cuda_runtime.h>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {
    // Write code here
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N)
         C[thread_id] = A[thread_id] - B[thread_id];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_sub<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}