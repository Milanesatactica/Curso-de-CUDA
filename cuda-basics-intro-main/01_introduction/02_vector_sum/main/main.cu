#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

void initialize_array_with_global_id(
    const unsigned int elements,
    unsigned int* h_array_id) {

    for (unsigned int i = 0; i < elements; ++i) {
        h_array_id[i] = i;
    }
}

void two_vec_addition(
    const unsigned int elements,
    const unsigned int* h_a,
    const unsigned int* h_b,
    unsigned int* h_c) {

    for (unsigned int i = 0; i < elements; ++i) {
        h_c[i] = h_a[i] + h_b[i];
    }
}

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

// main function (Host)
int main() {

    constexpr unsigned int elements = 1 << 10;
    constexpr size_t elements_size = elements * sizeof(unsigned int);

    unsigned int *h_a;
    unsigned int *h_b;
    unsigned int *h_c;
    cudaMallocHost(&h_a, elements_size);
    cudaMallocHost(&h_b, elements_size);
    cudaMallocHost(&h_c, elements_size);

    initialize_array_with_global_id( elements, h_a);
    initialize_array_with_global_id( elements, h_b);

    two_vec_addition( elements, h_a, h_b, h_c);

    for (unsigned int i = 0; i < elements; ++i) {

        printf("c[%u] = %u, expected: %u\n", i, h_c[i], 2*i );
        assert(h_c[i] ==2* i);
    }

// TODO 1: Uncomment and run the device code
//
//         1-a) What is the device currently doing?
//         1-b) Discuss the way to execute the sum on the device
//         1-c) Where does the data (arrays) should be to be used by
//              two_vec_addition_kernel?
/*
    // Set host vector to 0 for fair comparison 
    for (unsigned int i = 0; i < elements; ++i) {
        h_c[i] = 0;
    }

    // Device memory allocation
    unsigned int *d_a;
    unsigned int *d_b;
    unsigned int *d_c;
    cudaMalloc(&d_a, elements_size);
    cudaMalloc(&d_b, elements_size);
    cudaMalloc(&d_c, elements_size);

//    // 1D thread-block size 
//    constexpr unsigned int threads = 1 << 5;
//    dim3 thread_block(threads);
//
//    // Allocating enough threads to cover the array
//    const unsigned int blocks = (elements + threads - 1) / threads;
//
//    // Copy elements from ??? to ???
//    cudaMemcpy(h_a, d_a, elements_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(d_a, h_a, elements_size, cudaMemcpyHostToDevice);

    // Checking that the device array is correctly set
    for (unsigned int i = 0; i < elements; ++i) {
        assert(h_c[i] == 0);
    }

    // Device memory free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
*/

    // Host memory free
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}

