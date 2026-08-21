# CUDA Matrix Transpose

This experiment studies global-memory coalescing, shared-memory
bank conflicts and shared-memory padding using matrix-transpose
kernels.

The implementation is a source-faithful adaptation of NVIDIA's
CUDA matrix-transpose sample:

https://github.com/NVIDIA/cuda-samples/blob/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/6_Performance/transpose/transpose.cu

## Hardware

- GPU: NVIDIA GeForce GTX 1650
- Compute capability: 7.5
- Compilation target: `sm_75`
- Streaming multiprocessors: 16
- Warp size: 32 threads
- Global memory: 4.00 GiB
- Shared memory per block: 48 KiB

## Experiment configuration

- Matrix dimensions: 1024 × 1024
- Matrix elements: 1,048,576 FP32 values
- Tile dimensions: 32 × 32
- Thread-block dimensions: 32 × 16
- Threads per block: 512

Four kernels were compared:

1. `copy` — coalesced global-memory copy used as a performance baseline.
2. `transposeNaive` — coalesced reads but strided, uncoalesced writes.
3. `transposeCoalesced` — shared-memory tiling makes global reads and
   writes coalesced, but its 32 × 32 shared tile causes bank conflicts.
4. `transposeNoBankConflicts` — changes the shared tile from 32 × 32
   to 32 × 33, eliminating the bank conflicts.

## Build

The program was compiled with optimization, CUDA source-line
information and the native architecture target:

```bat
nvcc -std=c++17 -O2 -lineinfo -arch=sm_75 transpose_compare.cu -o transpose_compare.exe