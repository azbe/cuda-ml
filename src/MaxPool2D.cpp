#include <algorithm>
#include <cassert>
#include <cfloat>

#include "MaxPool2D.h"

MaxPool2D::MaxPool2D(int height, int width) :
	height(height),
	width(width)
{}

void MaxPool2D::runCPU(const Tensor * input_t, Tensor * output_t) const
{
	const Tensor3D* input = dynamic_cast<const Tensor3D*>(input_t);
	Tensor3D* output = dynamic_cast<Tensor3D*>(output_t);

	assert(output->getHeight() == int(input->getHeight() / this->height));
	assert(output->getWidth() == int(input->getWidth() / this->width));
	assert(output->getChannels() == input->getChannels());

	output->setData(-FLT_MAX);
	const int H = input->getHeight();
	const int W = input->getWidth();
	const int C = input->getChannels();

	const int OH = output->getHeight();
	const int OW = output->getWidth();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	int oidx = 0;
	for (int h = 0; h < OH; ++h) {
		for (int w = 0; w < OW; ++w) {
			for (int c = 0; c < C; ++c) {
				for (int kh = 0; kh < this->height; ++kh) {
					for (int kw = 0; kw < this->width; ++kw) {
						const int iidx = (h * this->height + kh) * W * C + (w * this->width + kw) * C + c;
						output_ptr[oidx] = std::max(output_ptr[oidx], input_ptr[iidx]);
					}
				}
				++oidx;
			}
		}
	}
}

void MaxPool2D::runGPU(const Tensor* input, Tensor* output) const
{
}
