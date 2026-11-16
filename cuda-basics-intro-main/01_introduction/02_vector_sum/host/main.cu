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

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}

