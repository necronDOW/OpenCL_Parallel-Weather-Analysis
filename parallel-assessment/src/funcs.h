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
	// Convert from a floating point array to an integer array.
	int* new_arr = new int[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] * multiplier;

	return new_arr;
}
fp_type* convert(int* arr, size_t size, int multiplier)
{
	// Convert from an integer array to a floating point array.
	fp_type* new_arr = new fp_type[size];
	for (size_t i = 0; i < size; i++)
		new_arr[i] = arr[i] / multiplier;

	return new_arr;
}

template<typename T>
T mean(T value, fp_type size)
{
	// Calculate the mean of a given value.
	return value / size;
}

template<typename T>
T source(const T* arr, int size, fp_type indexer)
{
	// Source an array value from an index as a percentage of the total size.
	int src_index = size * indexer;
	return arr[src_index];
}

template<typename T>
void Resize(T*& arr, size_t& size, int add_size)
{
	if (add_size == 0)
		return;

	// Claculate the new size for the array and create a buffer at the given size.
	size_t new_size = size + add_size;
	T* new_arr = new T[new_size];

	// Parse the values from the old array to the new one, whether the sizing is smaller or larger.
	memcpy(new_arr, arr, ((add_size < 0) ? new_size : size) * sizeof(T));
	for (int i = size; i < new_size; i++)
		new_arr[i] = 0;

	// Set the reference variables to the new values.
	size = new_size;
	arr = new_arr;
}

template<typename T>
bool Sorted(T*& arr, size_t size)
{
	// Sequentially check whether or not an array is sorted. Maximum complexity = O(N);
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
bool wg_size_changed = true;						// Whether the workgroup size was changed since last execution.
bool max_wg_size = false;							// Whether or not the work groups are max size.
size_t local_size;									// The currently selected work group size.
ProfilingResolution profiler_resolution = PROF_NS;	// The desired profiler resolution.

enum OptimizeFlags
{
	Performance,
	Precision
};
OptimizeFlags optimize_flag = Performance;			// The current optimization mode for the program.

void PrintProfilerInfo(std::string kernel_id, size_t ex_time, unsigned long* profiled_info, size_t ex_time_total = 0)
{
	// Output the detailed kernel execution information along side chronos based elapsed time for sequential executions.
	const char* resolution_str = GetResolutionString(profiler_resolution);
	std::string profiling_str = (profiled_info) ? GetFullProfilingInfo(profiled_info) : std::to_string(ex_time);
	std::string total_execution_str = std::to_string(ex_time_total);

	std::string output = "Kernel (" + kernel_id + ") execution time " + resolution_str + ": " + profiling_str 
		+ ((!ex_time_total) ? "" : "\nTotal execution time " + std::string(resolution_str) + ": " + total_execution_str);
	std::cout << output << std::endl;

	// Also write the resultant profiling info to a log file.
	winstr::Write(output.c_str());
}

template<typename T>
void ProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t arr_size, T*& arr, size_t len, const char* kernel_name)
{
	// Enqueue the kernel and read back the result, providing necessary profiling information.
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, arr_size, &arr[0]);

	// Print the profiling information for this kernel execution.
	unsigned long ex_time_total = timer::Stop(profiler_resolution);
	unsigned long ex_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	PrintProfilerInfo(kernel_name, ex_time, GetFullProfilingInfoData(prof_event, profiler_resolution), ex_time_total);

	// Flush the queue.
	queue.flush();
}

template<typename T>
void CumulativeProfiledExecution(cl::Kernel kernel, cl::Buffer buffer, size_t arr_size, T*& arr, size_t len, unsigned long& ex_time_total, unsigned long& ex_time, unsigned long* profiled_info)
{
	// Enqueue the kernel and read back the result, providing necessary profiling information.
	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len), cl::NDRange(local_size), NULL, &prof_event);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, arr_size, &arr[0]);

	// Add the profiling info to the three seperate profiling statistics.
	ex_time_total += timer::QuerySinceLast(profiler_resolution);
	ex_time += prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	unsigned long* this_profiled_info = GetFullProfilingInfoData(prof_event, profiler_resolution);
	for (int i = 0; i < 4; i++)
		profiled_info[i] += this_profiled_info[i];

	// Flush the queue.
	queue.flush();
}

template<typename T>
void CLResize(cl::Kernel kernel, T*& arr, size_t& size)
{
	// Calculate the best work group size for the device and return the min group size or max group size based on current settings.
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t pref_size = (max_wg_size) ? kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)
		: kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

	// Calculate and apply the padding size to the array.
	size_t padding_size = size % pref_size;
	if (padding_size)
		Resize(arr, size, pref_size - padding_size);

	// Set the local_size to the new prefered size.
	local_size = pref_size;
}

template<typename T>
void CheckResize(cl::Kernel kernel, T*& arr, size_t& size, size_t original_size)
{
	// If the work group size was changed wince last execution, resize the array by calling CLResize.
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
	// Create a new cl buffer with the provided memory mode and data size (in bytes).
	cl::Buffer buffer(context, mem_mode, data_size);
	
	// Enqueue the buffer differently based on whether or not the mem_mode is READ_ONLY or not.
	if (mem_mode == CL_MEM_READ_ONLY)
		queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data_size, &data[0]);
	else queue.enqueueFillBuffer(buffer, 0, 0, data_size);

	// Set the kernel argument at the given index to the new buffer and return the buffer.
	kernel.setArg(arg_index, buffer);
	return buffer;
}

// String concatination for kernels based on typeid's.
template<typename T> void ConcatKernelID(T type, std::string& original) { original += "_INVALID"; }
template<> void ConcatKernelID(int type, std::string& original) { original += "_INT"; }
template<> void ConcatKernelID(float type, std::string& original) { original += "_FP"; }
template<> void ConcatKernelID(double type, std::string& original) { original += "_FP"; }

template<typename T>
void Sum(T*& inbuf, T*& outbuf, size_t& len, size_t original_len)
{
	// Determine the kernel name using the type T, e.g. T == int will concatinate  "_INT".
	std::string kernel_id = "reduce_sum";
	ConcatKernelID(*inbuf, kernel_id);

	// Start a chrono timer and create the kernel with the determined id.
	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	// Check if the data set is in need of a resize, this will only resize if the local_size has changed since last execution.
	CheckResize(kernel, inbuf, len, original_len);

	// Reset outbuf to a blank array of type T.
	outbuf = new T[1]{ 0 };

	// Determine the byte size of outbuf and inbuf, and provide necessary kernel arguments for reduce_sum.
	size_t data_size = len * sizeof(T);
	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));

	// Output the profiled execution times for this particular kernel.
	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());
}

template<typename T>
void LocalMinMax(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, bool dir)
{
	// Determine the kernel name using the type T, e.g. T == int will concatinate  "_INT".
	std::string kernel_id = (dir) ? "reduce_max" : "reduce_min";
	ConcatKernelID(*inbuf, kernel_id);

	// Start a chrono timer and create the kernel with the determined id.
	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	// Check if the data set is in need of a resize, this will only resize if the local_size has changed since last execution.
	CheckResize(kernel, inbuf, len, original_len);

	// Reset outbuf to a blank array of type T.
	outbuf = new T[1] { 0 };

	// Determine the byte size of outbuf and inbuf, and provide necessary kernel arguments for reduce_max/min.
	size_t data_size = len * sizeof(T);
	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));

	// Output the profiled execution times for this particular kernel.
	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());
}

template<typename T>
void GlobalMinMax(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, bool dir)
{
	// Determine the kernel name using the type T, e.g. T == int will concatinate  "_INT".
	std::string kernel_id = (dir) ? "reduce_max_global" : "reduce_min_global";
	ConcatKernelID(*inbuf, kernel_id);

	// Start a chrono timer and create the kernel with the determined id.
	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	// Check if the data set is in need of a resize, this will only resize if the local_size has changed since last execution.
	CheckResize(kernel, inbuf, len, original_len);

	// Reset outbuf to a blank array of type T.
	outbuf = new T[len];

	// Determine the byte size of outbuf and inbuf, and provide necessary kernel arguments for reduce_max/min_global.
	size_t data_size = len * sizeof(T);
	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, data_size);

	// Output the profiled execution times for this particular kernel.
	ProfiledExecution(kernel, buffer_B, data_size, outbuf, len, kernel_id.c_str());
}

template<typename T>
void Variance(T*& inbuf, T*& outbuf, size_t& len, size_t original_len, T mean)
{
	// Determine the kernel name using the type T, e.g. T == int will concatinate  "_INT".
	std::string kernel_id = "sum_sqr_diff";
	ConcatKernelID(*inbuf, kernel_id);

	// Start a chrono timer and create the kernel with the determined id.
	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	// Check if the data set is in need of a resize, this will only resize if the local_size has changed since last execution.
	CheckResize(kernel, inbuf, len, original_len);

	// Reset outbuf to a blank array of type T.
	outbuf = new T[1] { 0 };

	// Determine the byte size of outbuf and inbuf, and provide necessary kernel arguments for sum_sqr_diff.
	size_t data_size = len * sizeof(T);
	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, sizeof(T));
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));
	kernel.setArg(3, mean);

	// Output the profiled execution times for this particular kernel.
	ProfiledExecution(kernel, buffer_B, sizeof(T), outbuf, len, kernel_id.c_str());

	// Return the mean of the sum of squared differences.
	outbuf[0] /= len;
}

template<typename T>
T* Sort(T*& inbuf, T outbuf[], size_t& len, size_t original_len)
{
	// Declare unsigned long variables for the profiling output.
	unsigned long ex_time_total = 0, ex_time = 0, profiled_info[4] { 0, 0, 0, 0 };

	// Determine the kernel name using the type T, e.g. T == int will concatinate  "_INT".
	std::string kernel_id = "bitonic_local";
	ConcatKernelID(*inbuf, kernel_id);

	// Start a chrono timer and create the kernel with the determined id.
	timer::Start();
	cl::Kernel kernel = cl::Kernel(program, kernel_id.c_str());

	// Check if the data set is in need of a resize, this will only resize if the local_size has changed since last execution.
	CheckResize(kernel, inbuf, len, original_len);

	// Reset outbuf to a blank array of type T.
	outbuf = new T[len];

	// Determine the byte size of outbuf and inbuf, and provide necessary kernel arguments for bitonic_local.
	size_t data_size = len * sizeof(T);
	cl::Buffer buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, inbuf, data_size);
	cl::Buffer buffer_B = EnqueueBuffer(kernel, 1, CL_MEM_READ_WRITE, outbuf, data_size);
	kernel.setArg(2, cl::Local(local_size * sizeof(T)));
	kernel.setArg(3, 0); // 0 represents an unshifted sort, thus when local_size = 32, sorting 0 -> 31, 32 -> 63, etc ...

	// Perform cumulative execution, this will accumulate the total execution time in the last 3 parameters ready for later use.
	CumulativeProfiledExecution(kernel, buffer_B, data_size, outbuf, len, ex_time_total, ex_time, profiled_info);

	int i = 1; // 1 represents an shifted sort, thus when local_size = 32, sorting 16 -> 47, 70 -> 101, etc ...
	while (!Sorted(outbuf, len))
	{
		// Only buffer_A and the merge arguments need to be re-assigned.
		buffer_A = EnqueueBuffer(kernel, 0, CL_MEM_READ_ONLY, outbuf, data_size);
		kernel.setArg(3, (i++ % 2));

		// Perform cumulative execution.
		CumulativeProfiledExecution(kernel, buffer_B, data_size, outbuf, len, ex_time_total, ex_time, profiled_info);
	}

	// Once Sorted() == true, the while loop is terminated and the cumulative profiler info is printed.
	PrintProfilerInfo(kernel_id, ex_time, profiled_info, ex_time_total);
	std::cout << "\n";
	timer::Stop();

	// Return the result to the function.
	return outbuf;
}

int* sorted_array_int = nullptr;
int* SortOptim(int*& A, int*& B, size_t& base_size, size_t original_size)
{
	// If the cached integer array is not set, perform 'shifted bitonic sort' on the data and store result within the cache.
	if (!sorted_array_int)
		sorted_array_int = Sort(A, B, base_size, original_size);

	// Return the cached integer array.
	return sorted_array_int;
}

fp_type* sorted_array_fp = nullptr;
fp_type* SortOptim(fp_type*& A, fp_type*& B, size_t& base_size, size_t original_size)
{
	// If the cached floating point array is not set, perform 'shifted bitonic sort' on the data and store result within the cache.
	if (!sorted_array_fp)
		sorted_array_fp = Sort(A, B, base_size, original_size);

	// Return the cached floating point array.
	return sorted_array_fp;
}

#endif