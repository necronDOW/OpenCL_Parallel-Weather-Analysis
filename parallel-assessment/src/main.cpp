#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>

#include "Utils.h"
#include "windows_fileread.h"
#include "analytics.h"
#include "funcs.h"

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
	queue = cl::CommandQueue(context);
	cl::Program::Sources sources;

	AddSources(sources, "my_kernels3.cl");
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

	std::cout << "Sequential Read (milliseconds): " << timer::QueryMilliseconds() << std::endl;

	out_size = winstr::query_line_count(inFile, len);
	out_arr = winstr::parse_lines(inFile, len, ' ', 5, out_size);

	std::cout << "Sequential Parse (milliseconds): " << timer::QueryMillisecondsSinceLast() << std::endl;

	local_size = 10;
	size_t padding_size = out_size % local_size;

	if (padding_size)
		resize(out_arr, out_size, padding_size);

	std::cout << "Sequential Resize (milliseconds): " << timer::QueryMillisecondsSinceLast() << "\n" << std::endl;
	timer::Stop();
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
		init_cl(platform_id, device_id);

		double* init_A;
		size_t padded_size = 0;
		init_data("./data/temp_lincolnshire.txt", init_A, padded_size);

		data_type* A = convert(init_A, padded_size, 10);
		data_type* B = new data_type[padded_size];

		size_t input_size = padded_size * sizeof(data_type);
		size_t output_size = padded_size * sizeof(data_type);
		size_t num_groups = padded_size / local_size;

		reduce_add_1(A, input_size, B, output_size, padded_size);
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("pause");
	return 0;
}
