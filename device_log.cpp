#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>


void inspect_devices() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA capable devices found\n";
    }

    std::stringstream buffer;

    for (int device_index = 0; device_index < deviceCount; device_index++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_index);

        buffer << time(NULL) << "\n"
            << "name: " << prop.name << "\n"
            << "l2CacheSize: " << prop.l2CacheSize << "\n"
            << "maxGridSize: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << "\n"
            << "maxThreadsDim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << "\n"
            << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n"
            << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n"
            << "multiProcessorCount: " << prop.multiProcessorCount << "\n"
            << "sharedMemPerBlock: " << prop.sharedMemPerBlock << "\n"
            << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << "\n"
            << "totalConstMem: " << prop.totalConstMem << "\n"
            << "totalGlobalMem: " << prop.totalGlobalMem << "\n"
            << "warpSize: " << prop.warpSize << "\n\n";
    }

    std::ofstream file("deviceProp.txt", std::ios::app);
    if (file) {
        file << buffer.str();
        file.close();
        std::cout << "CUDA capable devices found, 'deviceProp.txt' file updated\n";
    }
    else {
        std::cerr << "Failed to open deviceProp.txt\n";
    }
}