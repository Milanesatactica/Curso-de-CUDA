* * #include <cassert>
#include <iostream>

#include <cuda_runtime.h>

void initialize_array_with_global_id(
    const unsigned int elements,
    unsigned int* h_array_id) {

    for (unsigned int i = 0; i < elements; ++i) {
        h_array_id[i] = i;
    }
}

__global__ void initialize_array_with_global_id_kernel(const unsigned int elements , unsigned int* d_array_id) {

	const unsigned int tid =  blockIdx.x * blockDim.x + threadIdx.x;
	d_array_id[tid] = tid;
}


int main() {

    constexpr unsigned int elements = 1 << 10;

    constexpr size_t elements_size = elements * sizeof(unsigned int);

    unsigned int *h_array_id;

    cudaMallocHost(&h_array_id, elements_size);

    initialize_array_with_global_id( elements, h_array_id);

    for (unsigned int i = 0; i < elements; ++i) {
        assert(h_array_id[i] == i);
}

    unsigned int *d_array_id;
    cudaMalloc(&d_array_id, elements_size);

    constexpr unsigned int threads = 32;
    dim3 thread_block(threads);

      const unsigned int blocks = elements / threads;
//    const unsigned int blocks = (elements + threads - 1) / threads;

    initialize_array_with_global_id_kernel<<<blocks, thread_block>>>(elements, d_array_id);

    cudaMemcpy(h_array_id, d_array_id, elements_size,
        cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < elements; ++i) {
        assert(h_array_id[i] == i);
    }

    cudaFree(d_array_id);


    cudaFreeHost(h_array_id);

    return 0;
}

