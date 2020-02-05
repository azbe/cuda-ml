#include <cassert>
#include <cstring>

#include "Dense.h"

Dense::Dense(int prev_units, int num_units)
{
	this->weights = new Tensor2D(prev_units, num_units);
	this->bias = new Tensor1D(num_units);
}

void Dense::runCPU(const Tensor* input_t, Tensor* output_t) const
{
	const Tensor1D* input = dynamic_cast<const Tensor1D*>(input_t);
	Tensor1D* output = dynamic_cast<Tensor1D*>(output_t);
	const Tensor2D* weights = dynamic_cast<const Tensor2D*>(this->weights);
	const Tensor1D* bias = dynamic_cast<const Tensor1D*>(this->bias);

	assert(input->getSize() == weights->getHeight());
	assert(output->getSize() == weights->getWidth());

	output->setData(0.0);
	const int N = input->getSize();
	const int M = output->getSize();

	const float* input_ptr = input->getData();
	float* output_ptr = output->getData();
	const float* weights_ptr = weights->getData();
	const float* bias_ptr = bias->getData();
	for (int ocol = 0; ocol < M; ++ocol) {
		for (int icol = 0; icol < N; ++icol) {
			output_ptr[ocol] += input_ptr[icol] * weights_ptr[icol * M + ocol];
		}
		output_ptr[ocol] += bias_ptr[ocol];
	}
}
