#include <stdio.h>
#include <iostream>


__global__ void fill_vector_kernel(unsigned int* vector, const unsigned int elements) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < elements){

	 vector[idx] = idx;
	}
}

inline void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {

    constexpr unsigned int elements = 1<<20;

    constexpr size_t vector_size = elements*sizeof(unsigned int);

    unsigned int* d_vector;
    unsigned int* h_vector;
    check_cuda_error( cudaMalloc(&d_vector, vector_size) ); 
    cudaMalloc(&d_vector, vector_size);
    cudaMallocHost(&h_vector, vector_size);

    constexpr unsigned int threads =256;
    dim3 thread_block(threads);
    const unsigned int blocks  = (elements + threads - 1)/threads ;

    fill_vector_kernel<<<blocks,thread_block>>>(d_vector, elements);

    check_cuda_error( cudaMemcpy(h_vector, d_vector, vector_size, cudaMemcpyDeviceToHost) );
    cudaMemcpy(h_vector, d_vector, vector_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++)
        printf("vec[%d]=%d\n", i, h_vector[i]);

    for(int i=elements-10; i<elements; i++)
        printf("vec[%d]=%d\n", i, h_vector[i]);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_vector);
    cudaFreeHost(d_vector);
}
