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

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
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

#ifdef _DEBUG
		std::cout << "Build_Mode=Debug" << std::endl;
#else
		std::cout << "Build_Mode=Release" << std::endl;
#endif

		timer::Start();
		unsigned int len;
		const char* inFile = winstr::read_optimal("./data/temp_lincolnshire.txt", len);
		std::cout << "\nRead Time (milliseconds): " << timer::QueryMilliseconds() << std::endl;

		double* out = winstr::parse_lines(inFile, len, ' ', 5, 1873106);
		std::cout << "\nParse Time (milliseconds): " << timer::QueryMilliseconds() << std::endl;
		timer::Stop();

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

		typedef int mytype;
		std::vector<mytype> A(10, 1);

		size_t local_size = 10;

		size_t padding_size = A.size() % local_size;

		if (padding_size)
		{
			std::vector<int> A_ext(local_size - padding_size, 0);
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();
		size_t input_size = A.size() * sizeof(mytype);
		size_t nr_groups = input_elements / local_size;

		std::vector<mytype> B(input_elements);
		size_t output_size = B.size() * sizeof(mytype);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_1");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("pause");
	return 0;
}
