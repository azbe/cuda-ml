#include <cassert>

#include "gpu_util.h"
#include "Dense.h"

__global__ void denseKernel(const float* dev_input, const float* dev_weights, const float* dev_bias, float* dev_output, int N, int M)
{
	extern __shared__ float shared[];
	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;
	const int gidx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = tidx; i < N; i += blockDim.x) 
	{
		shared[i] = dev_input[i];
	}

	__syncthreads();

	if (gidx < M) 
	{
		float sum = 0.0;
		for (int i = 0; i < N; ++i) {
			sum += shared[i] * dev_weights[i * M + gidx];
		}
		dev_output[gidx] = sum + dev_bias[gidx];
	}
}

void Dense::runGPU(const Tensor* input_t, Tensor* output_t) const
{
	const Tensor1D* input = dynamic_cast<const Tensor1D*>(input_t);
	Tensor1D* output = dynamic_cast<Tensor1D*>(output_t);
	const Tensor2D* weights = dynamic_cast<const Tensor2D*>(this->weights);
	const Tensor1D* bias = dynamic_cast<const Tensor1D*>(this->bias);

	assert(input->getSize() == weights->getHeight());
	assert(output->getSize() == weights->getWidth());

	const int N = input->getSize();
	const int M = output->getSize();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	const float* weights_ptr = weights->getData();
	const float* bias_ptr = bias->getData();

	float* dev_input = 0;
	float* dev_weights = 0;
	float* dev_bias = 0;
	float* dev_output = 0;
	COPY_TO_DEVICE(input_ptr, input->getSize(), dev_input);
	COPY_TO_DEVICE(weights_ptr, weights->getSize(), dev_weights);
	COPY_TO_DEVICE(bias_ptr, bias->getSize(), dev_bias);
	COPY_TO_DEVICE(output_ptr, output->getSize(), dev_output);

	const int NUM_THREADS = gpu_util::MAX_THREADS;
	const int NUM_BLOCKS = gpu_util::CEIL(M, NUM_THREADS);
	denseKernel<<<NUM_BLOCKS, NUM_THREADS, N * sizeof(float)>>>(dev_input, dev_weights, dev_bias, dev_output, N, M);
	cudaMemcpy(output_ptr, dev_output, output->getSize() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(dev_weights);
	cudaFree(dev_bias);
	cudaFree(dev_output);
}


