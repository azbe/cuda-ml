#pragma once
#include "TrainableLayer.h"
#include "Tensors.h"

class Dense : public TrainableLayer
{
public:
	Dense() = delete;
	Dense(int prev_units, int num_units);
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
};

