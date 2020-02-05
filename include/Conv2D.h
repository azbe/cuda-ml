#pragma once
#include "TrainableLayer.h"
#include "Tensors.h"

class Conv2D : public TrainableLayer
{
public:
	Conv2D() = delete;
	Conv2D(int filters, int height, int width, int channels);
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
};

