#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device = 0;  // Device ID
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Block Size: " << prop.maxThreadsPerBlock << std::endl;

    return 0;
}