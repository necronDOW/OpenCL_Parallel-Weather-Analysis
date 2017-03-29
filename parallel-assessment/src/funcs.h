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

#include "windows_fileread.h"

typedef int PRECISION;

// ------------------------------------------------------------------------ Helper Functions ------------------------------------------------------------------------ //

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

template<typename T>
double mean(T value, double size)
{
	return value / size;
}

void Resize(PRECISION*& arr, size_t& size, size_t add_size)
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
bool max_wg_size = false;
size_t local_size;
ProfilingResolution profiler_resolution = PROF_NS;

void PrintProfilerInfo(std::string kernel_id, size_t ex_time, const cl::Event* event = nullptr)
{
	const char* resolution_str = GetResolutionString(profiler_resolution);
	std::string profiling_str = (event) ? GetFullProfilingInfo(*event, profiler_resolution) : std::to_string(ex_time);

	std::string output = "Kernel (" + kernel_id + ") execution time " + resolution_str + ": " + profiling_str;
	std::cout << output << std::endl;
	winstr::Write(output.c_str());
}

template<typename T>
void ProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t outbuf_size, T*& outbuf, size_t len, const char* kernel_name)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outbuf_size, &outbuf[0]);

	long long ex_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	PrintProfilerInfo(kernel_name, ex_time, &prof_event);
}

size_t CLResize(cl::Kernel kernel, PRECISION*& arr, size_t& size)
{
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = (max_wg_size) ? kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)
		: kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

	size_t padding_size = size % pref_size;
	if (padding_size)
		Resize(arr, size, pref_size - padding_size);

	return pref_size;
}

template<typename T>
cl::Buffer EnqueueBuffer(cl::Kernel kernel, int arg_index, int mem_mode, T* data, size_t data_size)
{
	cl::Buffer buffer(context, mem_mode, data_size);
	
	if (mem_mode == CL_MEM_READ_ONLY)
		queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data_size, &data[0]);
	else queue.enqueueFillBuffer(buffer, 0, 0, data_size);

	kernel.setArg(arg_index, buffer);
	return buffer;
}

void Sum(PRECISION inbuf[], PRECISION*& outbuf, size_t len)
{
	const char* kernel_id = "reduce_sum";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = CLResize(kernel, inbuf, len);
	outbuf = new PRECISION[1];
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(PRECISION));
	kernel.setArg(2, cl::Local(local_size * sizeof(PRECISION)));

	ProfiledExecution(kernel, buffer_B, sizeof(PRECISION), outbuf, len, kernel_id);
}

void LocalMinMax(PRECISION inbuf[], PRECISION*& outbuf, size_t len, bool dir)
{
	const char* kernel_id = (dir) ? "reduce_max" : "reduce_min";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = CLResize(kernel, inbuf, len);
	outbuf = new PRECISION[1];
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(PRECISION));
	kernel.setArg(2, cl::Local(local_size * sizeof(PRECISION)));

	ProfiledExecution(kernel, buffer_B, sizeof(PRECISION), outbuf, len, kernel_id);
}

void GlobalMinMax(PRECISION inbuf[], PRECISION*& outbuf, size_t len, bool dir)
{
	const char* kernel_id = (dir) ? "reduce_max_global" : "reduce_min_global";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = CLResize(kernel, inbuf, len);
	outbuf = new PRECISION[len];
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, data_size);

	ProfiledExecution(kernel, buffer_B, data_size, outbuf, len, kernel_id);
}

void Variance(PRECISION inbuf[], PRECISION*& outbuf, size_t len, int mean)
{
	const char* kernel_id = "sum_sqr_diff";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = CLResize(kernel, inbuf, len);
	outbuf = new PRECISION[1];
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(PRECISION));
	kernel.setArg(2, cl::Local(local_size * sizeof(PRECISION)));
	kernel.setArg(3, mean);

	ProfiledExecution(kernel, buffer_B, sizeof(PRECISION), outbuf, len, kernel_id);

	outbuf[0] /= len;
}

void BitonicSort(PRECISION*& inbuf, size_t len)
{
	const char* kernel_id = "sort_bitonic";
	cl::Kernel kernel = cl::Kernel(program, kernel_id);

	local_size = CLResize(kernel, inbuf, len);
	size_t data_size = len * sizeof(PRECISION);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);

	ProfiledExecution(kernel, buffer_A, data_size, inbuf, len, kernel_id);
}

#endif