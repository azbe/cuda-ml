#pragma once
#pragma once

#include "Tensors.h"

class Layer
{
public:
	virtual ~Layer();
	virtual void runCPU(const Tensor* input, Tensor* output) const = 0;
	virtual void runGPU(const Tensor* input, Tensor* output) const = 0;
};

