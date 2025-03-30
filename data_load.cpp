#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_load.h"
#include "device_log.h"
#include "constants.h"
#include <cstdlib>


gpuDataBatch::gpuDataBatch() {
    clear_batch();
}

gpuDataBatch::~gpuDataBatch() {
    if (batch.features) cudaFree(batch.features);
    if (batch.target) cudaFree(batch.target);
}

gpuDataBatch::gpuDataBatch(gpuDataBatch&& other) noexcept {
    batch = other.batch;
    other.batch.features = nullptr;
    other.batch.target = nullptr;
}

gpuDataBatch& gpuDataBatch::operator=(gpuDataBatch&& other) noexcept {
    if (this != &other) {
        clear_batch();

        batch = other.batch;

        other.batch.features = nullptr;
        other.batch.target = nullptr;
        other.batch.n_sample = 0;
        other.batch.n_feature = 0;
    }
    return *this;
}

void gpuDataBatch::clear_batch() {
    if (batch.features) cudaFree(batch.features);
    if (batch.target) cudaFree(batch.target);

    batch.features = nullptr;
    batch.target = nullptr;
    batch.n_sample = 0;
    batch.n_feature = 0;
}


DataLoader::DataLoader(
    const char* filepath,
    size_t batch_size,
    size_t n_feature
) :
    filepath_(filepath),
    batch_size_(batch_size),
    n_feature_(n_feature)
{
    file_ = fopen(filepath, "r");
    if (!file_) {
        fprintf(stderr, "Error, can't open file: %s \n", filepath);
        exit(EXIT_FAILURE);
    }
}

DataLoader::~DataLoader() {
    if (!file_) fclose(file_);
}

bool DataLoader::load_next_batch(gpuDataBatch& gpu_batch) {
    // allocate host memory
    float* host_features = (float*)malloc(batch_size_ * n_feature_ * sizeof(float));
    float* host_target = (float*)malloc(batch_size_ * sizeof(float));

    log_start_batch_allocation(constants::LOG_FILEPATH);

    size_t sample_index = 0;
    // read features to host memory
    for (; sample_index < batch_size_; ++sample_index) {
        for (size_t feature_index = 0; feature_index < n_feature_; ++feature_index) {
            if (fscanf(file_, "%f, ", host_features[sample_index * n_feature_ + feature_index]) != 1) {
                break;
            }
        }
        // read target to host memory
        if (fscanf(file_, "%f\n", host_target[sample_index]) != 1) {
            break;
        }
    }

    // if there is no data left in file — return false
    if (sample_index == 0) {
        free(host_features);
        free(host_target);
        return false;
    }

    // allocate device memory
    size_t feature_size = sample_index * n_feature_ * sizeof(float);
    size_t target_size = sample_index * sizeof(float);

    gpu_batch.clear_batch(); // free memory of current batch
    cudaMalloc(&gpu_batch.batch.features, feature_size);
    cudaMalloc(&gpu_batch.batch.target, target_size);

    // Copy memory from host -> device
    cudaMemcpy(
        gpu_batch.batch.features,
        host_features,
        feature_size,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        gpu_batch.batch.target,
        host_target,
        target_size,
        cudaMemcpyHostToDevice
    );

    gpu_batch.batch.n_sample = sample_index;
    gpu_batch.batch.n_feature = n_feature_;

    free(host_features);
    free(host_target);

    log_end_batch_allocation(
        constants::LOG_FILEPATH,
        gpu_batch.batch,
        feature_size,
        target_size
    );

    return true;
}
