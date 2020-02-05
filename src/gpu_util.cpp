#include "gpu_util.h"

namespace gpu_util
{
	void init() 
	{
		cudaError_t cudaStatus = cudaSetDevice(DEVICE);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice error: %s %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
			exit(EXIT_ERROR_CODE);
		}
	}

}; // namespace gpu_util