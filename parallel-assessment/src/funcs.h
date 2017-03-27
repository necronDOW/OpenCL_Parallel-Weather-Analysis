#ifndef funcs_h
#define funcs_h

#include <iostream>

#ifndef cl_included
	#define cl_included
	#ifdef __APPLE__
		#include <OpenCL/cl.hpp>
	#else
		#include <CL/cl.hpp>
	#endif
#endif

typedef int data_type;

int* convert(double* arr, size_t size, int multiplier)
{
	int* new_arr = new int[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] * multiplier;

	return new_arr;
}
double* convert(int* arr, size_t size, int multiplier)
{
	double* new_arr = new double[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] / multiplier;

	return new_arr;
}

void resize(double* arr, size_t& old_size, size_t add_size)
{
	size_t new_size = old_size + add_size;
	double* new_arr = new double[new_size];

	memcpy(new_arr, arr, old_size * sizeof(double));

	old_size = new_size;
	arr = new_arr;
}

// ------------------------------------------------------------------------ Parallel Functions ------------------------------------------------------------------------ //

cl::Context context;
cl::CommandQueue queue;
cl::Program program;
size_t local_size;

void reduce_add_1(data_type* inbuf, size_t inbuf_size, data_type* outbuf, size_t outbuf_size, size_t len)
{
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, inbuf_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, outbuf_size);

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, inbuf_size, &inbuf[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, outbuf_size);

	cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_2");
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_B);

	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size));

	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, outbuf_size, &outbuf[0]);
}

#endif