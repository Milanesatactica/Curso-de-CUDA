#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_from_thread_kernel()
{
    printf("Hello from thread %u, block %u with dim %u\n",
        threadIdx.x, blockIdx.x, blockDim.x );
}

void hello_from_thread()
{
    printf("Hello world from CPU\n");
}

int main()
{
    hello_from_thread();

    hello_from_thread_kernel<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}
