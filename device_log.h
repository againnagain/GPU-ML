#ifndef DEVICE_LOG_H
#define DEVICE_LOG_H

#include "data_load.h"

void inspect_devices(const char *filepath);
void log_start_batch_allocation(const char* filepath);
void log_end_batch_allocation(
    const char* filepath,
    const DataBatch& batch,
    size_t feature_size,
    size_t target_size
);
#endif