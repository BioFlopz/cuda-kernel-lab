#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << ": " << cudaGetErrorString(error) << '\n';         \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    constexpr int n = 2050;
    constexpr int threads_per_block = 256;

    const std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);
    std::vector<float> h_reference(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
        h_reference[i] = h_a[i] + h_b[i];
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (n + (threads_per_block - 1)) / threads_per_block;

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_milliseconds = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_milliseconds,
        start,
        stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(
        h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float maximum_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float error = std::fabs(h_c[i] - h_reference[i]);
        maximum_error = std::max(maximum_error, error);

        if (error > 1.0e-5f) {
            if (mismatches < 5) {
                std::cerr << "Mismatch at " << i
                          << ": GPU=" << h_c[i]
                          << ", CPU=" << h_reference[i] << '\n';
            }

            ++mismatches;
        }
    }

    const double bytes_read = 2.0 * static_cast<double>(n) * sizeof(float);

    const double bytes_written = static_cast<double>(n) * sizeof(float);

    const double elapsed_seconds = static_cast<double>(elapsed_milliseconds) / 1000.0;

    const double effective_bandwidth_gbps = ((bytes_read + bytes_written) / 1.0e9) / elapsed_seconds;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    const int total_threads = blocks * threads_per_block;

    std::cout << "Elements:          " << n << '\n'
              << "Blocks:            " << blocks << '\n'
              << "Threads per block: " << threads_per_block << '\n'
              << "Total threads:     " << total_threads << '\n'
              << "Maximum error:     " << maximum_error << '\n'
              << "Verification:      "
              << (mismatches == 0 ? "PASS" : "FAIL") << '\n';

    std::cout << "Kernel time:        "
              << elapsed_milliseconds << " ms\n"
              << "Bytes read:         "
              << bytes_read << '\n'
              << "Bytes written:      "
              << bytes_written << '\n'
              << "Effective bandwidth: "
              << effective_bandwidth_gbps << " GB/s\n";

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}