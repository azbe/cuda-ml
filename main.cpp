#include <cassert>
#include <cstdio>

#include "gpu_util.h"
#include "Runtime.h"
#include "tests.h"

int main(int argc, char** argv) {
	// CPU tests
	printf("=== CPU tests\n");
	printf("Dense: %s\n", tests::testDenseCPU() ? "PASSED" : "FAILED");
	printf("Conv2D: %s\n", tests::testConv2DCPU() ? "PASSED" : "FAILED");
	printf("MaxPool2D: %s\n", tests::testMaxPool2DCPU() ? "PASSED" : "FAILED");
	printf("GlobalAveragePool2D: %s\n", tests::testGlobalAveragePool2DCPU() ? "PASSED" : "FAILED");
	printf("Softmax: %s\n", tests::testSoftmax() ? "PASSED" : "FAILED");
	printf("Relu: %s\n", tests::testRelu() ? "PASSED" : "FAILED");

	// GPU tests
	gpu_util::init();
	printf("\n=== GPU tests\n");
	printf("Dense: %s\n", tests::testDenseGPU() ? "PASSED" : "FAILED");
	printf("Conv2D: %s\n", tests::testConv2DGPU() ? "PASSED" : "FAILED");

	return 0;
}