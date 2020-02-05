#include <cassert>
#include <cstdio>

#include "gpu_util.h"
#include "Conv2D.h"

__global__ void conv2dKernel(const float* dev_input, const float* dev_weights, const float* dev_bias, float* dev_output, int H, int W, int C, int N, int KH, int KW, int OH, int OW)
{
	extern __shared__ float shared[];
	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;
	const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int shsz = KW * KH * C;

	const int kidx = (gidx % N) * shsz;
	for (int i = tidx; i < shsz; i += blockDim.x) 
	{
		shared[i] = dev_weights[kidx + i];
	}

	__syncthreads();

	if (gidx < OH * OW * N)
	{
		const int oh = gidx / (OW * N);
		const int ow = (gidx / N) % OW;
		int sidx = oh * W * C + ow * C;
		float sum = 0.0;
		for (int i = 0; i < shsz; ++i)
		{
			const int row = (i / (KW * C));
			sum += dev_input[sidx + row * W * C + i] * shared[i];
		}
		dev_output[gidx] = sum + dev_bias[gidx];
	}
}

void Conv2D::runGPU(const Tensor* input_t, Tensor* output_t) const
{
	const Tensor3D* input = dynamic_cast<const Tensor3D*>(input_t);
	Tensor3D* output = dynamic_cast<Tensor3D*>(output_t);
	const Tensor4D* weights = dynamic_cast<const Tensor4D*>(this->weights);
	const Tensor1D* bias = dynamic_cast<const Tensor1D*>(this->bias);

	assert(input->getChannels() == weights->getChannels());
	assert(output->getHeight() == input->getHeight() - weights->getHeight() + 1);
	assert(output->getWidth() == input->getWidth() - weights->getWidth() + 1);
	assert(output->getChannels() == weights->getBatch());

	output->setData(0.0);
	const int H = input->getHeight();
	const int W = input->getWidth();
	const int C = input->getChannels();

	const int N = weights->getBatch();
	const int KH = weights->getHeight();
	const int KW = weights->getWidth();

	const int OH = output->getHeight();
	const int OW = output->getWidth();

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
	const int NUM_BLOCKS = gpu_util::CEIL(OH * OW * N, NUM_THREADS);
	conv2dKernel<<<NUM_BLOCKS, NUM_THREADS, KH * KW * C * sizeof(float)>>>(dev_input, dev_weights, dev_bias, dev_output, H, W, C, N, KH, KW, OH, OW);
	cudaMemcpy(output_ptr, dev_output, output->getSize() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(dev_weights);
	cudaFree(dev_bias);
	cudaFree(dev_output);
}