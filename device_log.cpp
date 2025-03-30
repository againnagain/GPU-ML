#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include "data_load.h"


void inspect_devices(const char* filepath) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA capable devices found\n";
    }

    std::stringstream buffer;

    for (int device_index = 0; device_index < deviceCount; device_index++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_index);

        time_t timestamp = time(NULL);
        struct tm datetime = *localtime(&timestamp);

        buffer << asctime(&datetime)
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

    std::ofstream file(filepath, std::ios::app);
    if (file) {
        file << buffer.str();
        file.close();
        std::cout << "CUDA capable devices found, 'deviceProp.txt' file updated\n";
    }
    else {
        std::cerr << "Failed to open deviceProp.txt\n";
    }
}

void log_start_batch_allocation(const char* filepath) {
    time_t timestamp = time(NULL);
    struct tm datetime = *localtime(&timestamp);

    std::stringstream buffer;
    buffer << "start batch allocation at " << asctime(&datetime) << "\n";

    std::ofstream file(filepath, std::ios::app);
    if (file) {
        file << buffer.str();
        file.close();
    }
    else {
        std::cerr << "Failed to open log file\n";
    }
}

void log_end_batch_allocation(
    const char* filepath,
    const DataBatch& batch,
    size_t feature_size,
    size_t target_size
) {
    time_t timestamp = time(NULL);
    struct tm datetime = *localtime(&timestamp);

    std::stringstream buffer;
    buffer << "end batch allocation at " << asctime(&datetime) << "\n"
        << "GPU memory allocated:" << "\n"
        << "     Features ptr: " << batch.features << " ("
        << feature_size / (1024.0f * 1024.0f) << "MB)" << "\n"
        << "     Target ptr: " << batch.target << " ("
        << target_size / (1024.0f * 1024.0f) << "MB)" << "\n";

    std::ofstream file(filepath, std::ios::app);
    if (file) {
        file << buffer.str();
        file.close();
    }
    else {
        std::cerr << "Failed to open log file\n";
    }
}