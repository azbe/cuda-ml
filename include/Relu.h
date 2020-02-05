#pragma once
#include "Layer.h"

class Relu : public Layer
{
public:
	Relu();
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
};

