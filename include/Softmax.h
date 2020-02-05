#pragma once
#include "Layer.h"

class Softmax : public Layer
{
public:
	Softmax();
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
};

