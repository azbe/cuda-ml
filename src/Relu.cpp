#include <algorithm>
#include <cassert>

#include "Relu.h"

Relu::Relu()
{}

void Relu::runCPU(const Tensor * input_t, Tensor * output_t) const
{
	const Tensor1D* input = dynamic_cast<const Tensor1D*>(input_t);
	Tensor1D* output = dynamic_cast<Tensor1D*>(output_t);

	assert(input->getSize() == output->getSize());

	output->setData(0.0);
	const int N = input->getSize();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	for (int i = 0; i < N; ++i) {
		output_ptr[i] = std::max(input_ptr[i], 0.0f);
	}
}

void Relu::runGPU(const Tensor * input, Tensor * output) const
{
	assert(0);
}
