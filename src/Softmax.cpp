#include <cassert>
#include <cmath>

#include "Softmax.h"

Softmax::Softmax()
{}

void Softmax::runCPU(const Tensor* input_t, Tensor* output_t) const
{
	const Tensor1D* input = dynamic_cast<const Tensor1D*>(input_t);
	Tensor1D* output = dynamic_cast<Tensor1D*>(output_t);

	assert(input->getSize() == output->getSize());

	output->setData(0.0);
	const int N = input->getSize();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	float sum = 0.0;
	for (int i = 0; i < N; ++i) {
		sum += exp(input_ptr[i]);
	}
	for (int i = 0; i < N; ++i) {
		output_ptr[i] = exp(input_ptr[i]) / sum;
	}
}

void Softmax::runGPU(const Tensor* input, Tensor* output) const
{
	assert(0);
}
