#include "TrainableLayer.h"

TrainableLayer::~TrainableLayer()
{
	delete weights;
	delete bias;
}

void TrainableLayer::setWeights(const float* data)
{
	weights->setData(data);
}

void TrainableLayer::setWeights(float value)
{
	weights->setData(value);
}

void TrainableLayer::setWeights(const char* path)
{
	weights->setData(path);
}

void TrainableLayer::setBias(const float* data)
{
	bias->setData(data);
}

void TrainableLayer::setBias(float value)
{
	bias->setData(value);
}

void TrainableLayer::setBias(const char* path)
{
	bias->setData(path);
}
