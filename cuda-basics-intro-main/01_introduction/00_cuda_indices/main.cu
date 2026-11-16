#include <iostream>

#include <cuda_runtime.h>

__global__ void print_threads_id_kernel() {

    // 1D output for cleaner visualization
//    printf("block-%u, thread-%u\n", blockIdx.x, threadIdx.x);


    // 3D output
    printf("block(%u, %u, %u) thread(%u, %u, %u)\n",
	blockIdx.x, blockIdx.y, blockIdx.z,
	threadIdx.x, threadIdx.y, threadIdx.z);

}

int main() {

    constexpr unsigned int threadsx = 8;
    constexpr unsigned int threadsy = 4;
    constexpr unsigned int threadsz = 5;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const unsigned int max_threads_per_block = prop.maxThreadsPerBlock;
    printf("Max thread-block size: %u\n", max_threads_per_block);

//	dim3 thread_block(threads);
//	dim3 thread_block(threads, threads);
	dim3 thread_block(threadsx, threadsy, threadsz);

    print_threads_id_kernel<<<1, thread_block>>>();
    cudaDeviceSynchronize();

    return 0;
}

