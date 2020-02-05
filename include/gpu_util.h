#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define COPY_TO_DEVICE(iptr, size, optr) \
	{ \
		cudaMalloc((void**) &optr, size * sizeof(float)); \
		cudaMemcpy(optr, iptr, size * sizeof(float), cudaMemcpyHostToDevice); \
	} 

namespace gpu_util
{
	static const int DEVICE = 0;
	static const int EXIT_ERROR_CODE = 14;
	static const int MAX_THREADS = 128;
	constexpr int CEIL(int x, int y) { return (x + y - 1) / y; }

	void init();

}; // namespace gpu_util
