#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Write code here
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    float x = input[thread_id];

    output[thread_id] = (__expf(x) - __expf(-x)) / (__expf(x) + __expf(-x));
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}