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

int* convert(fp_type* arr, size_t size, int multiplier)
{
	int* new_arr = new int[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] * multiplier;

	return new_arr;
}
fp_type* convert(int* arr, size_t size, int multiplier)
{
	fp_type* new_arr = new fp_type[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] / multiplier;

	return new_arr;
}

template<typename T>
T mean(T value, float size)
{
	return value / size;
}

template<typename T>
T source(const T* arr, int size, float indexer)
{
	int src_index = size * indexer;
	return arr[src_index];
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

template<typename T>
bool Sorted(T*& arr, size_t size)
{
	for (int i = 0; i < size - 1; i++)
	{
		if (arr[i] > arr[i+1])
			return false;
	}

	return true;
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

void PrintProfilerInfo(std::string kernel_id, size_t ex_time, unsigned long* profiled_info, size_t ex_time_total = 0)
{
	const char* resolution_str = GetResolutionString(profiler_resolution);
	std::string profiling_str = (profiled_info) ? GetFullProfilingInfo(profiled_info) : std::to_string(ex_time);
	std::string total_execution_str = std::to_string(ex_time_total);

	std::string output = "Kernel (" + kernel_id + ") execution time " + resolution_str + ": " + profiling_str 
		+ ((!ex_time_total) ? "" : "\nTotal execution time " + std::string(resolution_str) + ": " + total_execution_str);
	std::cout << output << std::endl;
	winstr::Write(output.c_str());
}

template<typename T>
void ProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t arr_size, T*& arr, size_t len, const char* kernel_name)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, arr_size, &arr[0]);

	unsigned long ex_time_total = timer::Stop(profiler_resolution);
	unsigned long ex_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	PrintProfilerInfo(kernel_name, ex_time, GetFullProfilingInfoData(prof_event, profiler_resolution), ex_time_total);
}

template<typename T>
void CumulativeProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t arr_size, T*& arr, size_t len, unsigned long& ex_time_total, unsigned long& ex_time, unsigned long* profiled_info)
{
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, arr_size, &arr[0]);

	ex_time_total += timer::QuerySinceLast(profiler_resolution);
	ex_time += prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	unsigned long* this_profiled_info = GetFullProfilingInfoData(prof_event, profiler_resolution);
	for (int i = 0; i < 4; i++)
		profiled_info[i] += this_profiled_info[i];
}

template<typename T>
void CLResize(cl::Kernel kernel, T*& arr, size_t& size)
{
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = (max_wg_size) ? kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)
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
		std::cout << "Resizing Array ... ";

		Resize(arr, size, original_size-size);
		CLResize(kernel, arr, size);
		wg_size_changed = false;

		std::cout << "Done\n";
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
template<> void ConcatKernelID(float type, std::string& original) { original += "_FP"; }

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
T* Sort(T*& inbuf, T*& outbuf, size_t& len, size_t original_len)
{
	unsigned long ex_time_total = 0, ex_time = 0, profiled_info[4] { 0, 0, 0, 0 };

	std::string kernel_id = "bitonic_local";
	ConcatKernelID(*inbuf, kernel_id);

	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	CheckResize(kernel, inbuf, len, original_len);

	outbuf = new T[len];
	size_t data_size = len * sizeof(T);

	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, data_size);
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));
	kernel.setArg(3, 0);

	CumulativeProfiledExecution(kernel, buffer_B, data_size, outbuf, len, ex_time_total, ex_time, profiled_info);

	int i = 1;
	while (!Sorted(outbuf, len))
	{
		buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, outbuf, data_size);
		kernel.setArg(3, (i++ % 2));

		CumulativeProfiledExecution(kernel, buffer_B, data_size, outbuf, len, ex_time_total, ex_time, profiled_info);
	}

	PrintProfilerInfo(kernel_id, ex_time, profiled_info, ex_time_total);
	std::cout << "\n";
	timer::Stop();

	T* results = new T[original_len];
	for (int i = 0; i < original_len; i++)
		results[i] = outbuf[i];

	return results;
}

int* sorted_array_int = nullptr;
int* SortOptim(int*& A, int*& B, size_t& base_size, size_t original_size)
{
	if (!sorted_array_int)
		sorted_array_int = Sort(A, B, base_size, original_size);

	return sorted_array_int;
}

fp_type* sorted_array_fp = nullptr;
fp_type* SortOptim(fp_type*& A, fp_type*& B, size_t& base_size, size_t original_size)
{
	if (!sorted_array_fp)
		sorted_array_fp = Sort(A, B, base_size, original_size);

	return sorted_array_fp;
}

#endif