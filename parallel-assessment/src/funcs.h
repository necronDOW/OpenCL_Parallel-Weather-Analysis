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

typedef int PRECISION;

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

void resize(PRECISION*& arr, size_t& size, size_t add_size)
{
	size_t new_size = size + add_size;
	PRECISION* new_arr = new PRECISION[new_size];

	memcpy(new_arr, arr, size * sizeof(PRECISION));
	for (int i = size; i < new_size; i++)
		new_arr[i] = 0;

	size = new_size;
	arr = new_arr;
}

// ------------------------------------------------------------------------ Parallel Functions ------------------------------------------------------------------------ //

cl::Context context;
cl::CommandQueue queue;
cl::Program program;

size_t local_size;
long long profiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t outbuf_size, PRECISION*& outbuf, size_t len)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outbuf_size, &outbuf[0]);

	return prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
}

size_t cl_resize(cl::Kernel kernel, PRECISION*& arr, size_t& size)
{
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

	size_t padding_size = size % pref_size;
	if (padding_size)
		resize(arr, size, pref_size - padding_size);

	return pref_size;
}

void reduceAdd1(PRECISION inbuf[], PRECISION*& outbuf, size_t len)
{
	const char* kernel_id = "reduce_add_4";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = cl_resize(kernel, inbuf, len);
	outbuf = new PRECISION[len];
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, data_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, data_size);

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, data_size, &inbuf[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, data_size);

	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	kernel.setArg(2, cl::Local(data_size));

	size_t ex_time = profiledExecution(kernel, buffer_B, data_size, outbuf, len);
	std::cout << "Kernel (" << kernel_id << ") execution time [ns]: " << ex_time << std::endl;

	queue.finish();
}

#endif