#pragma once

class Tensor
{
public:
	virtual ~Tensor();
	float& operator[](int index);
	const float& operator[](int index) const;
	float* getData();
	const float* getData() const;
	void setData(const float* data);
	void setData(float value);
	void setData(const char* path);
	virtual int getSize() const = 0;

protected:
	float* data = nullptr;
};

class Tensor1D : public Tensor
{
public:
	Tensor1D(int size);
	int getSize() const;

private:
	int size = 0;
};

class Tensor2D : public Tensor
{
public:
	Tensor2D(int height, int width);
	int getSize() const;
	int getHeight() const;
	int getWidth() const;

private:
	int height = 0;
	int width = 0;
};

class Tensor3D : public Tensor
{
public:
	Tensor3D(int height, int width, int channels);
	int getSize() const;
	int getHeight() const;
	int getWidth() const;
	int getChannels() const;

private:
	int height = 0;
	int width = 0;
	int channels = 0;
};

class Tensor4D : public Tensor
{
public:
	Tensor4D(int batch, int height, int width, int channels);
	int getSize() const;
	int getBatch() const;
	int getHeight() const;
	int getWidth() const;
	int getChannels() const;

private:
	int batch = 0;
	int height = 0;
	int width = 0;
	int channels = 0;
};

