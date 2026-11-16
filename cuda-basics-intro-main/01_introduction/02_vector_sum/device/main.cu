#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

__global__ void initialize_array_with_global_id_kernel(
    const unsigned int elements,
    unsigned int* d_array_id) {

    const unsigned int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < elements) {
        d_array_id[tid] = tid;
    }
}

__global__ void two_vec_addition_kernel(
    const unsigned int elements,
    const unsigned int* d_a,
    const unsigned int* d_b,
    unsigned int* d_c) {

    const unsigned int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < elements) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

int main() {

    constexpr unsigned int elements = 1 << 10;
    constexpr size_t elements_size = elements * sizeof(unsigned int);


    // Device memory allocation
    unsigned int *d_a;
    unsigned int *d_b;
    unsigned int *d_c;
    unsigned int *h_c;
    cudaMalloc(&d_a, elements_size);
    cudaMalloc(&d_b, elements_size);
    cudaMalloc(&d_c, elements_size);
//    h_c = (unsigned int*)malloc(elements_size);
    cudaMallocHost(&h_c, elements_size);

    // 1D thread-block size 
    constexpr unsigned int threads = 1 << 5;
    dim3 thread_block(threads);

    // Allocating enough threads to cover the array
    const unsigned int blocks = (elements + threads - 1) / threads;

   initialize_array_with_global_id_kernel<<<blocks,thread_block>>>(elements, d_a);
   initialize_array_with_global_id_kernel<<<blocks,thread_block>>>(elements, d_b);

   two_vec_addition_kernel<<<blocks,thread_block>>>(elements, d_a, d_b, d_c);

   cudaMemcpy(h_c, d_c, elements_size, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < elements; ++i) {

        printf("c[%u] = %u, expected: %u\n", i, h_c[i], 2*i );
        assert(h_c[i] ==2* i);
    }


    // Device memory free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_c);
//    free(h_c);

    return 0;
}

