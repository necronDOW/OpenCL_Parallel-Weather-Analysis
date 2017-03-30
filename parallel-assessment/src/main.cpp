#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>

#include "Utils.h"
#include "windows_fileread.h"
#include "analytics.h"
#include "funcs.h"
#include "paths.h"
#include "menu_system.h"

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

void PrintHelp() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

inline void InitCL(int platform_id, int device_id)
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

inline void InitData(const char* dir, fp_type*& out_arr, size_t& out_size)
{
	timer::Start();
	unsigned int len;
	const char* inFile = winstr::ReadOptimal(dir, len);

	std::cout << "Sequential read "<< GetResolutionString(profiler_resolution) << ": " << timer::Query(profiler_resolution) << std::endl;

	out_size = winstr::QueryLineCount(inFile, len);
	out_arr = winstr::ParseLines(inFile, len, ' ', 5, out_size);

	std::cout << "Sequential parse " << GetResolutionString(profiler_resolution) << ": " << timer::QuerySinceLast(profiler_resolution) << std::endl;
	timer::Stop();
}

int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;
	char* file_dir = "temp_lincolnshire_short.txt";

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { PrintHelp(); }
		else if (strcmp(argv[i], "-s") == 0) { file_dir = "temp_lincolnshire_short.txt"; }
	}

	try
	{
		InitPaths();
		InitCL(platform_id, device_id);
		InitMenus();

		int *A, *B;
		fp_type *A_f, *B_f;

		size_t base_size = 0;
		InitData(std::string(data_path + file_dir).c_str(), A_f, base_size);

		size_t original_size = base_size;
		A = convert(A_f, base_size, 10);

		std::cout << std::endl;
		
		bool finished = false;
		while (!finished)
		{
			if (optimize_flag == Performance)
				MainMenu(A, B, base_size, original_size, finished);
			else if (optimize_flag == Precision)
				MainMenu(A_f, B_f, base_size, original_size, finished);
		}
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		system("pause");
	}

	return 0;
}
