#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(){
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    if(!deviceCount){
        printf("No devices found suppoting CUDA.\n");
    }
    else{
        printf("Detected %d CUDA capable device(s).\n", deviceCount);
    }

    FILE *fp;
    fp = fopen("lab_device_properties.txt","w");
    if(fp == NULL){
        printf("Error opening file \"lab_device_properties\".\n");
    }

    for(int i=0; i<deviceCount; i++){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        fprintf(fp, "\nDevice %d: %s \n", i, deviceProp.name);
        fprintf(fp, "  Total amount of global memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        fprintf(fp, "  CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(fp, "  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        fprintf(fp, "  Total amount of constant memory: %zu bytes\n", deviceProp.totalConstMem);
        fprintf(fp, "  Total amount of shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        fprintf(fp, "  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        fprintf(fp, "  Warp size: %d\n", deviceProp.warpSize);
        fprintf(fp, "  Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        fprintf(fp, "  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(fp, "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        fprintf(fp, "  Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        fprintf(fp, "  Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        fprintf(fp, "  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
    }

    fclose(fp);

    return 0;
}