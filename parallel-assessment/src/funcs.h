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
T mean(T value, double size)
{
	return value / size;
}

template<typename T>
void Resize(T*& arr, size_t& size, int add_size)
{
	if (add_size == 0)
		return;

	size_t new_size = size + add_size;
	T* new_arr = new T[new_size];

	memcpy(new_arr, arr, ((add_size < 0) ? new_size : size) * sizeof(T));
	for (int i = size; i < new_size; i++)
		new_arr[i] = 0;

	size = new_size;
	arr = new_arr;
}

// ------------------------------------------------------------------------ Parallel Functions ------------------------------------------------------------------------ //

cl::Context context;
cl::CommandQueue queue;
cl::Program program;
bool wg_size_changed = true;
bool max_wg_size = false;
size_t local_size;
ProfilingResolution profiler_resolution = PROF_NS;

enum OptimizeFlags
{
	Performance,
	Precision
};
OptimizeFlags optimize_flag = Performance;

void PrintProfilerInfo(std::string kernel_id, size_t ex_time, const cl::Event* event = nullptr, size_t ex_time_total = 0)
{
	const char* resolution_str = GetResolutionString(profiler_resolution);
	std::string profiling_str = (event) ? GetFullProfilingInfo(*event, profiler_resolution) : std::to_string(ex_time);
	std::string total_execution_str = std::to_string(ex_time_total);

	std::string output = "Kernel (" + kernel_id + ") execution time " + resolution_str + ": " + profiling_str 
		+ ((!ex_time_total) ? "" : "\nTotal execution time " + std::string(resolution_str) + ": " + total_execution_str);
	std::cout << output << std::endl;
	winstr::Write(output.c_str());
}

template<typename T>
void ProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t outbuf_size, T*& outbuf, size_t len, const char* kernel_name)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outbuf_size, &outbuf[0]);

	long long ex_time_total = timer::Stop(profiler_resolution);
	long long ex_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	PrintProfilerInfo(kernel_name, ex_time, &prof_event, ex_time_total);
}

template<typename T>
void CLResize(cl::Kernel kernel, T*& arr, size_t& size)
{
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = (max_wg_size) ? kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) / 2
		: kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

	size_t padding_size = size % pref_size;
	if (padding_size)
		Resize(arr, size, pref_size - padding_size);

	local_size = pref_size;
}

template<typename T>
void CheckResize(cl::Kernel kernel, T*& arr, size_t& size, size_t original_size)
{
	if (wg_size_changed)
	{
		std::cout << "Resizing Array ..." << std::endl;

		Resize(arr, size, original_size-size);
		CLResize(kernel, arr, size);
		wg_size_changed = false;
	}
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

template<typename T> void ConcatKernelID(T type, std::string& original) { original += "_INVALID"; }
template<> void ConcatKernelID(int type, std::string& original) { original += "_INT"; }
template<> void ConcatKernelID(double type, std::string& original) { original += "_DOUBLE"; }

template<typename T>
void Sum(T*& inbuf, T*& outbuf, size_t& len, size_t original_len)
{
	std::string kernel_id = "reduce_sum";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);
	outbuf = new T[1];
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));

	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());
}

template<typename T>
void LocalMinMax(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, bool dir)
{
	std::string kernel_id = (dir) ? "reduce_max" : "reduce_min";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);
	outbuf = new T[1];
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));

	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());
}

template<typename T>
void GlobalMinMax(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, bool dir)
{
	std::string kernel_id = (dir) ? "reduce_max_global" : "reduce_min_global";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);
	outbuf = new T[len];
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, data_size);

	ProfiledExecution(kernel, buffer_B, data_size, outbuf, len, kernel_id.c_str());
}

template<typename T>
void Variance(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, T mean)
{
	std::string kernel_id = "sum_sqr_diff";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);
	outbuf = new T[1];
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));
	kernel.setArg(3, mean);

	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());

	outbuf[0] /= len;
}

template<typename T>
void BitonicSort(T*& inbuf, size_t& len, size_t original_len)
{
	std::string kernel_id = "sort_bitonic";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);

	ProfiledExecution(kernel, buffer_A, data_size, inbuf, len, kernel_id.c_str());
}

#endif