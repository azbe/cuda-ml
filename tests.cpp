#include <cmath>
#include <cstdio>

#include "tests.h"

namespace tests
{
	bool compareResults(const float* output, const float* expected, int size) 
	{
		bool correct = true;
		for (int i = 0; i < size; ++i) {
			float diff = fabs(output[i] - expected[i]);
			correct &= (diff < EPSILON);
		}
		return correct;
	}

	bool testConv2DCPU()
	{
		Tensor3D input(28, 28, 16);
		input.setData("files/conv2d_input.bin");

		Tensor3D expected(26, 26, 32);
		expected.setData("files/conv2d_output.bin");

		Conv2D layer(32, 3, 3, 16);
		layer.setWeights("files/conv2d_weights.bin");
		layer.setBias("files/conv2d_bias.bin");

		Tensor3D output(26, 26, 32);
		layer.runCPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testDenseCPU() 
	{
		Tensor1D input(1024);
		input.setData("files/dense_input.bin");

		Tensor1D expected(2048);
		expected.setData("files/dense_output.bin");

		Dense layer(1024, 2048);
		layer.setWeights("files/dense_weights.bin");
		layer.setBias("files/dense_bias.bin");

		Tensor1D output(2048);
		layer.runCPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testMaxPool2DCPU()
	{
		Tensor3D input(14, 14, 32);
		input.setData("files/max_pool_input.bin");

		Tensor3D expected(7, 7, 32);
		expected.setData("files/max_pool_output.bin");

		MaxPool2D layer(2, 2);

		Tensor3D output(7, 7, 32);
		layer.runCPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testGlobalAveragePool2DCPU()
	{
		Tensor3D input(7, 7, 16);
		input.setData("files/avg_pool_input.bin");

		Tensor1D expected(16);
		expected.setData("files/avg_pool_output.bin");

		GlobalAveragePool2D layer;

		Tensor1D output(16);
		layer.runCPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testSoftmax()
	{
		Tensor1D input(10);
		input.setData("files/softmax_input.bin");

		Tensor1D expected(10);
		expected.setData("files/softmax_output.bin");

		Softmax layer;

		Tensor1D output(10);
		layer.runCPU(&input, &output);

		const float* output_ptr = output.getData();
		const float* expected_ptr = expected.getData();

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testRelu()
	{
		Tensor1D input(10);
		input.setData("files/relu_input.bin");

		Tensor1D expected(10);
		expected.setData("files/relu_output.bin");

		Relu layer;

		Tensor1D output(10);
		layer.runCPU(&input, &output);

		const float* output_ptr = output.getData();
		const float* expected_ptr = expected.getData();

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testDenseGPU()
	{
		Tensor1D input(1024);
		input.setData("files/dense_input.bin");

		Tensor1D expected(2048);
		expected.setData("files/dense_output.bin");

		Dense layer(1024, 2048);
		layer.setWeights("files/dense_weights.bin");
		layer.setBias("files/dense_bias.bin");

		Tensor1D output(2048);
		layer.runGPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

	bool testConv2DGPU()
	{
		Tensor3D input(28, 28, 16);
		input.setData("files/conv2d_input.bin");

		Tensor3D expected(26, 26, 32);
		expected.setData("files/conv2d_output.bin");

		Conv2D layer(32, 3, 3, 16);
		layer.setWeights("files/conv2d_weights.bin");
		layer.setBias("files/conv2d_bias.bin");

		Tensor3D output(26, 26, 32);
		layer.runGPU(&input, &output);

		return compareResults(output.getData(), expected.getData(), output.getSize());
	}

} // tests