#include <cassert>
#include <cstdio>

#include "Conv2D.h"

Conv2D::Conv2D(int filters, int height, int width, int channels)
{
	this->weights = new Tensor4D(filters, height, width, channels);
	this->bias = new Tensor1D(filters);
}

void Conv2D::runCPU(const Tensor* input_t, Tensor* output_t) const
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

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	const float* weights_ptr = weights->getData();
	const float* bias_ptr = bias->getData();
	for (int h = 0; h < H - KH + 1; ++h) {
		for (int w = 0; w < W - KW + 1; ++w) {
			for (int n = 0; n < N; ++n) {
				const int oidx = h * (W - KW + 1) * N + w * N + n;
				for (int kh = 0; kh < KH; ++kh) {
					for (int kw = 0; kw < KW; ++kw) {
						for (int c = 0; c < C; ++c) {
							const int iidx = (h + kh) * W * C + (w + kw) * C + c;
							const int widx = n * KH * KW * C + kh * KW * C + kw * C + c;
							output_ptr[oidx] += input_ptr[iidx] * weights_ptr[widx];
						}
					}
				}
				output_ptr[oidx] += bias_ptr[n];
			}
		}
	}
}