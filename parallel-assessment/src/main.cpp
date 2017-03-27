#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
#include "windows_fileread.h"
#include "analytics.h"

using namespace std::chrono;

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

typedef double data_type;

void resize(data_type* arr, size_t& old_size, size_t add_size)
{
	size_t new_size = old_size + add_size;
	data_type* new_arr = new data_type[new_size];

	memcpy(new_arr, arr, old_size * sizeof(data_type));

	old_size = new_size;
	arr = new_arr;
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
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context);

		cl::Program::Sources sources;
		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);


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

		timer::Start();
		unsigned int len;
		const char* inFile = winstr::read_optimal("./data/temp_lincolnshire.txt", len);
		std::cout << "Sequential Read (nanoseconds): " << timer::QueryNanoseconds() << std::endl;

		size_t size = winstr::query_line_count(inFile, len);
		data_type* A = winstr::parse_lines(inFile, len, ' ', 5, size);
		std::cout << "Sequential Parse (nanoseconds): " << timer::QueryNanosecondsSinceLast() << std::endl;

		size_t local_size = 10;
		size_t padding_size = size % local_size;

		if (padding_size)
			resize(A, size, padding_size);
		std::cout << "Sequential Resize (nanoseconds): " << timer::QueryNanosecondsSinceLast() << "\n" << std::endl;
		timer::Stop();

		data_type* B = new data_type[size];
		size_t input_size = size * sizeof(data_type);
		size_t output_size = size * sizeof(data_type);
		size_t num_groups = size / local_size;

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_1");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(size), cl::NDRange(local_size));

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		std::cout << B[0] << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("pause");
	return 0;
}
