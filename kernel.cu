
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "device_log.h"
#include "constants.h"

int main()
{
	inspect_devices(constants::LOG_FILEPATH);
}