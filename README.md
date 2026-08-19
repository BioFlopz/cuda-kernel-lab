# cuda-kernel-lab

# CUDA Kernel Lab

A hands-on CUDA programming and GPU performance-engineering project. The goal is to develop, verify, profile and optimize progressively more sophisticated CUDA kernels.

## Environment baseline

The CUDA toolchain successfully compiled and executed a device-query program on the following system:

| Property                  |                   Value |
| ------------------------- | ----------------------: |
| GPU                       | NVIDIA GeForce GTX 1650 |
| CUDA devices detected     |                       1 |
| Compute capability        |                     7.5 |
| Native compilation target |                 `sm_75` |
| Streaming multiprocessors |                      16 |
| Warp size                 |              32 threads |
| Maximum threads per block |                    1024 |
| Global memory             |                4.00 GiB |
| Shared memory per block   |                  48 KiB |
| Registers per block       |                  65,536 |
| Memory bus width          |                128 bits |

The environment program was compiled with:

```bat
nvcc -std=c++17 -O2 -arch=sm_75 device_info.cu -o device_info.exe
```

This establishes a working CUDA development baseline and confirms that `sm_75` is the appropriate architecture target for this GPU.

## Repository structure

* `00-environment/` — CUDA environment and device-capability verification
* `01-vector-add/` — first CUDA kernel, correctness checking and timing
* Future directories will contain increasingly advanced kernels and performance investigations.
