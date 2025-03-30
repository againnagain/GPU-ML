#ifndef DATA_LOAD_H
#define DATA_LOAD_H

#include <cstddef>
#include <stdio.h>
#include <vector>


struct DataBatch {
    float* features;
    float* target;
    size_t n_sample;
    size_t n_feature;
};

class gpuDataBatch {
public:
    DataBatch batch;

    gpuDataBatch();
    ~gpuDataBatch();

    // delete copy constructor to avoid dangling pointer
    gpuDataBatch(const gpuDataBatch&) = delete;
    gpuDataBatch& operator=(const gpuDataBatch&) = delete;

    /*
    move semantics;
    allows to use class object as return of the function (and to not create copy of this object)
    */
    gpuDataBatch(gpuDataBatch&& other) noexcept;
    gpuDataBatch& operator=(gpuDataBatch&& other) noexcept;

    void clear_batch();
};

class DataLoader {
public:
    DataLoader(const char* filepath, size_t batch_size, size_t n_feature);
    ~DataLoader();

    bool load_next_batch(gpuDataBatch& batch);

private:
    const char* filepath_;
    size_t batch_size_;
    size_t n_feature_;
    FILE* file_;
};

#endif
