#pragma once

#include "Layer.h"
#include "Tensors.h"

class TrainableLayer : public Layer
{
public:
	virtual ~TrainableLayer();
	void setWeights(const float* data);
	void setWeights(float value);
	void setWeights(const char* path);
	void setBias(const float* data);
	void setBias(float value);
	void setBias(const char* path);

protected:
	Tensor* weights;
	Tensor* bias;
};

