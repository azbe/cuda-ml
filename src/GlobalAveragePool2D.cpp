#include <cassert>

#include "GlobalAveragePool2D.h"

GlobalAveragePool2D::GlobalAveragePool2D()
{}

void GlobalAveragePool2D::runCPU(const Tensor* input_t, Tensor* output_t) const
{
	const Tensor3D* input = dynamic_cast<const Tensor3D*>(input_t);
	Tensor1D* output = dynamic_cast<Tensor1D*>(output_t);

	assert(input->getChannels() == output->getSize());

	output->setData(0.0);
	const int H = input->getHeight();
	const int W = input->getWidth();
	const int C = input->getChannels();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	for (int c = 0; c < C; ++c) {
		float moving_average = 0.0;
		int moving_count = 0;
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {
				moving_average = ((moving_count / float(moving_count + 1)) * moving_average) + (input_ptr[h * W * C + w * C + c] / (moving_count + 1));
				++moving_count;
			}
		}
		output_ptr[c] = moving_average;
	}
}

void GlobalAveragePool2D::runGPU(const Tensor* input, Tensor* output) const
{
	assert(0);
}
