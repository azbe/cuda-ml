#pragma once
#include "Layer.h"

class MaxPool2D : public Layer
{
public:
	MaxPool2D(int height, int width);
	void runCPU(const Tensor* input, Tensor* output) const override;
	void runGPU(const Tensor* input, Tensor* output) const override;
private:
	int height;
	int width;
};
