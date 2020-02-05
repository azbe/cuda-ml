#pragma once

#include "Dense.h"
#include "Conv2D.h"
#include "MaxPool2D.h"
#include "GlobalAveragePool2D.h"
#include "Relu.h"
#include "Softmax.h"
#include "Tensors.h"
#include "tests.h"

namespace tests
{
	static const float EPSILON = 0.001f;

	bool testConv2DCPU();
	bool testDenseCPU();
	bool testMaxPool2DCPU();
	bool testGlobalAveragePool2DCPU();
	bool testSoftmax();
	bool testRelu();

	bool testDenseGPU();
	bool testConv2DGPU();

} // namespace tests