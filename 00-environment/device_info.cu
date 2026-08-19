#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(                                                  \
                stderr,                                                    \
                "CUDA error at %s:%d: %s\n",                               \
                __FILE__,                                                  \
                __LINE__,                                                  \
                cudaGetErrorString(error));                                \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::printf("CUDA devices: %d\n\n", device_count);

    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::printf("Device %d: %s\n", device, properties.name);
        std::printf("Compute capability: %d.%d\n",
                    properties.major,
                    properties.minor);
        std::printf("Streaming multiprocessors: %d\n",
                    properties.multiProcessorCount);
        std::printf("Warp size: %d\n", properties.warpSize);
        std::printf("Maximum threads per block: %d\n",
                    properties.maxThreadsPerBlock);
        std::printf("Global memory: %.2f GiB\n",
                    properties.totalGlobalMem /
                        (1024.0 * 1024.0 * 1024.0));
        std::printf("Shared memory per block: %zu bytes\n",
                    properties.sharedMemPerBlock);
        std::printf("Registers per block: %d\n",
                    properties.regsPerBlock);
        std::printf("Memory bus width: %d bits\n",
                    properties.memoryBusWidth);
    }

    return 0;
}