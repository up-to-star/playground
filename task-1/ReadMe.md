
# Task
high performance gemm implementation on Nvidia A100
- use CUDA core (in fp32)
    - add your code in MMult_v0.cu
    - you can add more versions of matrix multiplication in MMult_v{number of version}.cu
- TensorCore (in fp16)
    - add your code in MMult_v0.cu
    - you can add more versions of matrix multiplication in MMult_v{number of version}.cu


# Usage

- start a docker image
  
  ```bash
  # image: torch-xla:r2.1.0-cuda12.1-cudnn8.9.6-playground
  # yourport: e.g. 21234
  docker run --gpus all --name your_docker_name -v /yourpath:/code -v /nvme/model_hub:/root/model_hub -it -p yourport:22 --entrypoint /bin/bash torch-xla:r2.1.0-cuda12.1-cudnn8.9.6-playground  
  cd /code
  git clone https://github.com/PJLAB-CHIP/playground.git
  cd playground/task-1
  ```
- revise *Makefile* to choose the **test precision** and **version of matrix multiplication**:
```makefile
TESTFILE := test_fp32 #format: test_{fp32/fp16}
MATMUL := MMult_v0 #format: MMult_v{number of version}/MMult_t_v{number of version}
```
- Run the program with the following command:
```bash
make run
#run "make -B run" instead if compiler can't detect changes in Makefile.
```
- Check test results in the directory *output_files*.
    - output file will be named as *output_{version}_{test_percision}.txt*
    - e.g. *output_MMult_v0_test_fp32.txt*

- test **cuBLAS**:
    Change Makefile:
    ```makefile
    TESTFILE := test_cuBLAS_cudacore_fp32   # or test_cuBLAS_tensorcore_fp16
    MATMUL := MMult_v0 # any available version is OK.
    ```
- benchmark cuda cublas performanc using torch
  ```bash
  python benchmark_cublas_use_torch.py
  ```
  ![image](https://github.com/PJLAB-CHIP/playground/assets/9277155/e6ef26aa-5ffc-4df5-9a74-942e3425db19)

# Example final results

## CUDA Core(FP32)
| Version | v0 | v1 | v2 | v3 | v4 | cuBLAS | Theory Peak |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| Average error | 0.011539 | 0.011532 | 0.011510 | 0.011657 | 0.011611 | / | / |
| TFLOPS | 2.415983 | 3.858012 | 9.240127 | 15.155111 | 17.163582 | 18.386241 | 19.5 |

## Tensor Core(FP16)

| Version | v0 | v1 | v2 | v3 | cuBLAS | Theory Peak |
| --- | --- | --- | --- | --- | --- | --- |
| Average error | 0.011781 | 0.011705 | 0.011730 | 0.001924 |0.015336 | / |
| TFLOPS | 18.097715 | 53.054460 | 59.355550 | 213.128590 |222.115580 | 312 |


# Notes
- Run *get_device_properties.cu* to check device properties
    - ```bash
      #example command
      nvcc -o get_device_properties get_device_properties.cu
      ./get_device_properties 
      ```
    - You will be able to check device properties in output file *lab_device_properties.txt*
    - These properties are helpful for further optimization
      
- **Pay attention**: check if GPU is in full utilization!
    - Check:
      ```bash
      nvidia-smi
      ```
      ![image](https://github.com/PJLAB-CHIP/playground/assets/93306395/91ca3296-4a7f-46d2-abed-3d51c9a66b36)

    - GPU0 is used by default. If it is in full utilization, TFLOPS measured will be about half of the performance measured in the ideal case. In this case, use other GPUs with the following command:
      ```bash
      export CUDA_VISIBLE_DEVICES={number of chosen GPU}
      ```
    - Run cuBLAS and test performance to verify that you have switched to a proper GPU.
 
- If codes are compiled with flag *-lineinfo* for debugging (In this way, your source CUDA C/C++ code will be able to be shown in Nsight Compute for locating problems), change settings in Makefile:
```makefile
DEBUG_FLAG := on #If compiled in normal mode, set "off" or any other words except "on" instead. 
```
- It is highly recommended that you use **Nsight Compute** to debug/ analyze the detailed running information of your kernel function/ guide your optimization......
  information about how to use **Nsight Compute** has been uploaded in FeiShu documents in "09. CUDA生态-04. playground-01. 高性能矩阵乘-3. Nsight Compute使用说明"

# Some Details
- Use **OpenBLAS** as reference to verify correctness
- About evaluation of algorithm
    - Method
        - Go through the flow for ten times:
          -  generate random matrices
          -  run the matrix multiplication process and calculate average error and TFLOPS for each process
          -  Accumulate evaluation indices
          -  Averaging
    - Calculation formulas
         > Average error = ( Σ( Σ(abs(my_res[i]-ref_res[i]) / ref_res[i]) ) ) / 10.0
         
         > TFLOPS = (flops_per_matmul \* 1.0e-12) / (msec_per_matmul / 1000.0)
         >> flops_per_matmul = 2.0\*m\*k\*n; msec_per_matmul = (Σruntime) / 10.0

# References

## CUDA Core


- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 2-3

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 4-5
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.2 Thread Block Tile: 利用 Shared Memory 减少重复访存

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 6-6.3 Thread coarsening
- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_4 & MMult_cuda_5
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370) Naive 实现的分析：到底差在哪里？

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 6-6.1 Memory coalescing, 6.2 Hiding memory latency
- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_9
- [cuda/MMult_cuda_9.cu](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_9.cu)
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370) 极致的访存优化
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.3 Warp Tile 与 Thread Tile: 利用寄存器消除 Shared Memory 瓶颈


- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_12
- [cuda/MMult_cuda_12.cu](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_12.cu)
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.4 Double Buffer: 让 GEMM 流水并行起来

## Tensor Core



- [cuda学习：学习nvcuda::wmma实现高效gemm](https://zhuanlan.zhihu.com/p/353208013) simple version



- [cuda学习：学习nvcuda::wmma实现高效gemm](https://zhuanlan.zhihu.com/p/353208013) sample version with detailed annotations
- [Official sample provided by NVIDIA](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)


- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098/) 4.5 提高L2 Cache命中率
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893) 调整线程块分配到的计算位置(swizzle)


Source code:
- [src/wmma/wmma_async_stage3.cu](https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/wmma/wmma_async_stage3.cu) 3 stages pipeline with WMMA API

Asynchronous data copy:
- [ Data Movement and Conversion Instructions: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) To know the usage of cp.async instructions
- [Performance Guidance for memcpy_async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) To know the usage of asynchronous data copy

Multi-buffer with prefetching:
- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098) 5 Pipeline优化-5.2 Stage
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893) 使用数据预取(prefetch)

Permute to use memory coalescing and avoid bank conflicts:
- [cuda（cutlass）编程之swizzle](https://www.bilibili.com/video/BV1Jb421e7UN/?spm_id_from=333.999.0.0&vd_source=2fe7991a33356057a2e41a2d37f9b7e0) A more detailed video explanation of swizzle based on CUTLASS

## For Further Study

- [基于 CUTE 的 GEMM 优化【1】—— Baseline 实现](https://zhuanlan.zhihu.com/p/695063154)
- [基于 CUTE 的 GEMM 优化【2】—— 高效 GEMM 实现，超越 Cublas 20%](https://zhuanlan.zhihu.com/p/696028389)
- [cute系列讲解](https://www.zhihu.com/people/reed-84-49/posts)
