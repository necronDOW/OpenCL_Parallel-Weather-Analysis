#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>

#include "Utils.h"
#include "windows_fileread.h"
#include "analytics.h"
#include "funcs.h"
#include "paths.h"

#ifndef cl_included
	#define cl_included
	#ifdef __APPLE__
		#include <OpenCL/cl.hpp>
	#else
		#include <CL/cl.hpp>
	#endif
#endif

//int welfords_variance(int data[], int size)
//{
//	int N = 0;
//	int mean = 0;
//	unsigned int M2 = 0;
//
//	for (int i = 0; i < size; i++)
//	{
//		N++;
//		int delta = data[i] - mean;
//		mean += (delta / N);
//		M2 += delta*delta;
//	}
//
//	return M2 / (N - 1);
//}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void init_cl(int platform_id, int device_id)
{
	context = GetContext(platform_id, device_id);
	queue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);
	cl::Program::Sources sources;

	AddSources(sources, "kernels.cl");
	program = cl::Program(context, sources);

	std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
	std::cout << analytics::BuildInfo() << std::endl;

	try
	{
		program.build();
	}
	catch (const cl::Error& err)
	{
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw err;
	}
}

void init_data(const char* dir, double*& out_arr, size_t& out_size)
{
	timer::Start();
	unsigned int len;
	const char* inFile = winstr::read_optimal(dir, len);

	std::cout << "Sequential read [ns]: " << timer::QueryNanoseconds() << std::endl;

	out_size = winstr::query_line_count(inFile, len);
	out_arr = winstr::parse_lines(inFile, len, ' ', 5, out_size);

	std::cout << "Sequential parse [ns]: " << timer::QueryNanosecondsSinceLast() << std::endl;
	timer::Stop();
}

void printResults(double _min, double _max, double _mean)
{
	printf("\nOpenCL Results:\n   Minimum: %.1f\n   Maximum: %.1f\n   Mean: %.5f\n\n", _min, _max, _mean);
}

template<typename T>
double mean(T value, double size)
{
	return value / size;
}

int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	try
	{
		init_paths();
		init_cl(platform_id, device_id);

		double* init_A;
		size_t base_size = 0;
		init_data(std::string(data_path + "temp_lincolnshire.txt").c_str(), init_A, base_size);

		size_t original_size = base_size;
		PRECISION* A = convert(init_A, base_size, 10);
		PRECISION* B = new PRECISION[base_size];

		std::cout << std::endl;
		
		reduce_minmax_global(A, B, base_size, false);
		double data_min = B[0] / 10.0;

		reduce_minmax_global(A, B, base_size, true);
		double data_max = B[0] / 10.0;

		reduce_sum(A, B, base_size);
		double data_mean = mean(B[0] / 10.0, original_size);

		printResults(data_min, data_max, data_mean);
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("pause");
	return 0;
}
