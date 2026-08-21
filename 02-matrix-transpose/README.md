# CUDA Matrix Transpose

This experiment studies global-memory coalescing, shared-memory bank conflicts
and shared-memory padding using matrix-transpose kernels.

The implementation is a source-faithful adaptation of NVIDIA's CUDA
matrix-transpose sample:

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

- Matrix dimensions: 1024 x 1024
- Matrix elements: 1,048,576 FP32 values
- Tile dimensions: 32 x 32
- Thread-block dimensions: 32 x 16
- Threads per block: 512

Four kernels were compared:

1. `copy` - coalesced global-memory copy used as a performance baseline.
2. `transposeNaive` - coalesced reads but strided, uncoalesced writes.
3. `transposeCoalesced` - shared-memory tiling makes global reads and writes
   coalesced, but its 32 x 32 shared tile causes bank conflicts.
4. `transposeNoBankConflicts` - changes the shared tile from 32 x 32 to
   32 x 33, eliminating the bank conflicts.

## Build

The program was compiled with optimization, CUDA source-line information and
the native architecture target:

```bat
nvcc -std=c++17 -O2 -lineinfo -arch=sm_75 transpose_compare.cu -o transpose_compare.exe
```

## Correctness validation

The program verifies every GPU result against a CPU reference.

Validation results:

- CPU verification: passed
- Compute Sanitizer memcheck: 0 errors
- Compute Sanitizer racecheck: 0 hazards
- Compute Sanitizer synccheck: 0 errors

Commands:

```bat
compute-sanitizer --tool memcheck transpose_compare.exe
compute-sanitizer --tool racecheck transpose_compare.exe
compute-sanitizer --tool synccheck transpose_compare.exe
```

## Benchmark results

| Kernel | Time (ms) | Effective bandwidth (GB/s) |
|---|---:|---:|
| Simple copy | 0.08785 | 88.9340 |
| Naive transpose | 0.22589 | 34.5847 |
| Coalesced transpose | 0.12853 | 60.7823 |
| Bank-conflict-free transpose | 0.08313 | 93.9812 |

The shared-memory coalesced kernel reached approximately 1.76 times the
bandwidth of the naive transpose.

Padding the shared tile increased bandwidth from 60.7823 GB/s to
93.9812 GB/s, a further 1.55-times improvement.

The final padded kernel reached approximately 2.72 times the bandwidth of the
naive transpose and performed comparably to the copy baseline.

## Bank-conflict explanation

Shared memory contains 32 banks, and successive 32-bit words map to successive
banks.

With an unpadded 32 x 32 tile, threads reading down one column access words
separated by a stride of 32:

```text
bank = (lane * 32 + fixed_column) mod 32
```

Every lane therefore maps to the same bank, producing a 32-way bank conflict.

Padding the second dimension changes the stride to 33:

```text
bank = (lane * 33 + fixed_column) mod 32
```

Because `33 mod 32 = 1`, successive lanes map to successive banks.

The padding increases shared-memory use from 4,096 bytes to 4,224 bytes, an
additional 128 bytes per block.

## Nsight Compute evidence

The kernels were profiled separately:

```bat
ncu --set full -k transposeCoalesced -s 1 -c 1 -f -o transpose_coalesced transpose_compare.exe

ncu --set full -k transposeNoBankConflicts -s 1 -c 1 -f -o transpose_no_bank_conflicts transpose_compare.exe
```

| Nsight Compute measurement | Unpadded | Padded |
|---|---:|---:|
| Shared-load bank conflicts | 1,015,808 | 0 |
| Shared-load wavefronts | 1,048,576 | 32,768 |
| Ideal shared-load wavefronts | 32,768 | 32,768 |
| Excessive shared wavefronts | 1,015,808 | 0 |
| Short-scoreboard stall | 27.25 | 2.05 |
| MIO-throttle stall | 60.81 | 1.99 |
| Kernel duration | 113.25 microseconds | 72.67 microseconds |
| DRAM throughput utilization | 56.42% | 87.15% |

The unpadded kernel generated 32 wavefronts for every shared-load request. The
padded kernel generated one wavefront per request and reported zero
shared-memory bank conflicts.

After eliminating the shared-memory bottleneck, DRAM throughput utilization
increased to 87.15%. The limiting resource therefore moved from conflicted
shared-memory accesses toward device-memory bandwidth.

## References

- CUDA C++ Best Practices Guide, Sections 10.2.3.1 and 10.2.3.3:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory
- NVIDIA CUDA matrix-transpose sample:
  https://github.com/NVIDIA/cuda-samples/blob/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/6_Performance/transpose/transpose.cu
- UIUC Profiling, Spring 2026, pages 54-56:
  https://lumetta.web.engr.illinois.edu/408-Sum26/slide-copies/profiling-from-S26.pdf#page=54
