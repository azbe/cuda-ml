#pragma once
#include "Layer.h"

class GlobalAveragePool2D : public Layer 
{
public:
	GlobalAveragePool2D();
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
};

