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

void resize(data_type* arr, size_t& size, size_t add_size)
{
	size_t new_size = size + add_size;
	data_type* new_arr = new data_type[new_size];

	memcpy(new_arr, arr, size * sizeof(data_type));

	size = new_size;
	arr = new_arr;
}

// ------------------------------------------------------------------------ Parallel Functions ------------------------------------------------------------------------ //

cl::Context context;
cl::CommandQueue queue;
cl::Program program;

size_t local_size;
long long profiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t outbuf_size, data_type*& outbuf, size_t len)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outbuf_size, &outbuf[0]);

	return prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
}

size_t cl_resize(cl::Kernel kernel, data_type* arr, size_t& size)
{
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

	size_t padding_size = size % pref_size;
	if (padding_size)
		resize(arr, size, pref_size - padding_size);

	return pref_size;
}

void reduceAdd1(data_type* inbuf, data_type* outbuf, size_t len)
{
	const char* kernel_id = "reduce_add_2";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = cl_resize(kernel, inbuf, len);
	size_t input_size = len * sizeof(data_type);

	cl_resize(kernel, outbuf, len);
	size_t output_size = len * sizeof(data_type);

	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &inbuf[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);

	size_t ex_time = profiledExecution(kernel, buffer_B, output_size, outbuf, len);
	std::cout << "Kernel (" << kernel_id << ") execution time [ns]: " << ex_time << std::endl;
}

#endif