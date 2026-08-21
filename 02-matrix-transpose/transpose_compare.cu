

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

// Selected from NVIDIA CUDA Samples Common/helper_cuda.h.
// static const char* _cudaGetErrorEnum(cudaError_t error)
// {
//     return cudaGetErrorName(error);
// }

// template <typename T>
// void check(T result, char const* const func, const char* const file,
//            int const line)
// {
//     if (result) {
//         std::fprintf(
//             stderr,
//             "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
//             file,
//             line,
//             static_cast<unsigned int>(result),
//             _cudaGetErrorEnum(result),
//             func);
//         std::exit(EXIT_FAILURE);
//     }
// }

// #define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << ": " << cudaGetErrorString(error) << '\n';         \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Selected from NVIDIA CUDA Samples Common/helper_image.h.
template <class T, class S>
inline bool compareData(const T* reference, const T* data,
                        const unsigned int len, const S epsilon,
                        const float threshold)
{
    assert(epsilon >= 0);

    bool result = true;
    unsigned int error_count = 0;

    for (unsigned int i = 0; i < len; ++i) {
        float diff =
            static_cast<float>(reference[i]) -
            static_cast<float>(data[i]);
        bool comp = (diff <= epsilon) && (diff >= -epsilon);
        result &= comp;
        error_count += !comp;
    }

    if (threshold == 0.0f) {
        return result;
    }

    if (error_count) {
        std::printf(
            "%4.2f(%%) of bytes mismatched (count=%u)\n",
            static_cast<float>(error_count) * 100 /
                static_cast<float>(len),
            error_count);
    }

    return len * threshold > error_count;
}

#define TILE_DIM 32
#define BLOCK_ROWS 16
#define NUM_REPS 100

int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;

__global__ void copy(float* odata, float* idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = xIndex + width * yIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index + i * width] = idata[index + i * width];
    }
}

__global__ void transposeNaive(float* odata, float* idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index_out + i] = idata[index_in + i * width];
    }
}

__global__ void transposeCoalesced(float* odata, float* idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = xIndex + yIndex * width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = xIndex + yIndex * height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeNoBankConflicts(float* odata, float* idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = xIndex + yIndex * width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = xIndex + yIndex * height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

void computeTransposeGold(float* gold, float* idata, const int size_x, const int size_y)
{
    for (int y = 0; y < size_y; ++y)
    {
        for (int x = 0; x < size_x; ++x)
        {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}

int main()
{
    const int size_x = MATRIX_SIZE_X;
    const int size_y = MATRIX_SIZE_Y;

    if (size_x != size_y)
    {
        std::printf("Matrix must be square.\n");
        return EXIT_FAILURE;
    }

    if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0)
    {
        std::printf("Matrix size must be an integral multiple of TILE_DIM.\n");
        return EXIT_FAILURE;
    }

    void (*kernel)(float*, float*, int, int);
    const char* kernelName;

    dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM);
    dim3 threads(TILE_DIM, BLOCK_ROWS);

    cudaEvent_t start;
    cudaEvent_t stop;

    size_t mem_size = static_cast<size_t>(sizeof(float) * size_x * size_y);

    float* h_idata = (float*)std::malloc(mem_size);
    float* h_odata = (float*)std::malloc(mem_size);
    float* transposeGold = (float*)std::malloc(mem_size);
    float* gold;

    float* d_idata;
    float* d_odata;

    CUDA_CHECK(cudaMalloc((void**)&d_idata, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&d_odata, mem_size));

    for (int i = 0; i < size_x * size_y; ++i)
    {
        h_idata[i] = static_cast<float>(i);
        h_odata[i] = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_odata, h_odata, mem_size, cudaMemcpyHostToDevice));

    computeTransposeGold(transposeGold, h_idata, size_x, size_y);

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf(
        "Matrix size: %dx%d, tile size: %dx%d, block size: %dx%d\n\n",
        size_x,
        size_y,
        TILE_DIM,
        TILE_DIM,
        TILE_DIM,
        BLOCK_ROWS);

    bool success = true;

    for (int k = 0; k < 4; ++k)
    {
        if (k == 0)
        {
            kernel = &copy;
            kernelName = "simple copy";
            gold = h_idata;
        }
        else if (k == 1)
        {
            kernel = &transposeNaive;
            kernelName = "naive transpose";
            gold = transposeGold;
        }
        else if (k == 2)
        {
            kernel = &transposeCoalesced;
            kernelName = "coalesced transpose";
            gold = transposeGold;
        }
        else
        {
            kernel = &transposeNoBankConflicts;
            kernelName = "bank-conflict-free transpose";
            gold = transposeGold;
        }

        CUDA_CHECK(cudaGetLastError());

        // One untimed warm-up launch.
        kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);

        CUDA_CHECK(cudaEventRecord(start, 0));

        for (int i = 0; i < NUM_REPS; ++i)
        {
            kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float kernelTime;
        CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

        CUDA_CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

        bool result = compareData(gold, h_odata, size_x * size_y, 0.01f, 0.0f);

        if (!result)
        {
            std::printf("*** %s kernel FAILED ***\n", kernelName);
            success = false;
        }

        float averageTime = kernelTime / NUM_REPS;


        float kernelBandwidth =
            2.0f * 1000.0f * mem_size /
            (1024 * 1024 * 1024) /
            averageTime;

        std::printf(
            "transpose %s, Throughput = %.4f GB/s, Time = %.5f ms, "
            "Size = %u fp32 elements, NumDevsUsed = %u, "
            "Workgroup = %u\n",
            kernelName,
            kernelBandwidth,
            averageTime,
            size_x * size_y,
            1,
            TILE_DIM * BLOCK_ROWS);

        for (int i = 0; i < size_x * size_y; ++i)
        {
            h_odata[i] = 0.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_odata, h_odata, mem_size, cudaMemcpyHostToDevice));
    }

    std::free(h_idata);
    std::free(h_odata);
    std::free(transposeGold);

    cudaFree(d_idata);
    cudaFree(d_odata);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (!success)
    {
        std::printf("Test failed!\n");
        return EXIT_FAILURE;
    }

    std::printf("Test passed\n");
    return EXIT_SUCCESS;
}
