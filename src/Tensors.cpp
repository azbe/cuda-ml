#include <cassert>
#include <cstdio>
#include <cstring>

#include "Tensors.h"

Tensor::~Tensor()
{
	if (this->data) {
		delete[] this->data;
	}
}

float& Tensor::operator[](int index)
{
	return this->data[index];
}

const float& Tensor::operator[](int index) const
{
	return this->data[index];
}

void Tensor::setData(const float* data)
{
	memcpy(this->data, data, this->getSize() * sizeof(float));
}

void Tensor::setData(float value)
{
	for (size_t i = 0; i < this->getSize(); ++i) {
		this->data[i] = value;
	}
}

void Tensor::setData(const char* path)
{
	FILE *fptr;
	fptr = fopen(path, "rb");
	if (fptr) {
		fread(this->data, sizeof(float), this->getSize(), fptr);
		fclose(fptr);
	}
}

float* Tensor::getData() 
{
	return this->data;
}

const float* Tensor::getData() const
{
	return this->data;
}

Tensor1D::Tensor1D(int size) :
	size(size)
{
	this->data = new float[size];
}

int Tensor1D::getSize() const
{
	return this->size;
}

Tensor2D::Tensor2D(int height, int width) :
	height(height), width(width)
{
	int size = height * width;
	this->data = new float[size];
}

int Tensor2D::getSize() const
{
	return this->height * this->width;
}

int Tensor2D::getHeight() const
{
	return this->height;
}

int Tensor2D::getWidth() const
{
	return this->width;
}

Tensor3D::Tensor3D(int height, int width, int channels) :
	height(height), width(width), channels(channels)
{
	int size = height * width * channels;
	this->data = new float[size];
}

int Tensor3D::getSize() const
{
	return this->height * this->width * this->channels;
}

int Tensor3D::getHeight() const
{
	return this->height;
}

int Tensor3D::getWidth() const
{
	return this->width;
}

int Tensor3D::getChannels() const
{
	return this->channels;
}

Tensor4D::Tensor4D(int batch, int height, int width, int channels) :
	batch(batch), height(height), width(width), channels(channels)
{
	int size = batch * height * width * channels;
	this->data = new float[size];
}

int Tensor4D::getSize() const
{
	return this->batch * this->height * this->width * this->channels;
}

int Tensor4D::getBatch() const
{
	return this->batch;
}

int Tensor4D::getHeight() const
{
	return this->height;
}

int Tensor4D::getWidth() const
{
	return this->width;
}

int Tensor4D::getChannels() const
{
	return this->channels;
}

