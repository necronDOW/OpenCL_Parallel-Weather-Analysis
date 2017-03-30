#ifndef utils_h
#define utils_h

#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include "paths.h"

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	if (!v.empty()) {
		out << '[';
		copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
		out << "\b\b]";
	}
	return out;
}

std::string GetPlatformName(int platform_id) {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	return platforms[platform_id].getInfo<CL_PLATFORM_NAME>();
}

std::string GetDeviceName(int platform_id, int device_id) {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector<cl::Device> devices;
	platforms[platform_id].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &devices);
	return devices[device_id].getInfo<CL_DEVICE_NAME>();
}

const char *getErrorString(cl_int error) {
	switch (error){
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

void CheckError(cl_int error) {
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << getErrorString(error) << std::endl;
		exit(1);
	}
}

void AddSources(cl::Program::Sources& sources, const std::string& file_name) {
	//TODO: add file existence check
	std::ifstream file(kernel_path + file_name);
	std::string* source_code = new std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	sources.push_back(std::make_pair((*source_code).c_str(), source_code->length() + 1));
}

std::string ListPlatformsDevices() {

	std::stringstream sstream;
	std::vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	sstream << "Found " << platforms.size() << " platform(s):" << std::endl;

	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		sstream << "\nPlatform " << i << ", " << platforms[i].getInfo<CL_PLATFORM_NAME>() << ", version: " << platforms[i].getInfo<CL_PLATFORM_VERSION>();

		sstream << ", vendor: " << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
		//		sstream << ", extensions: " << platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>() << endl;

		std::vector<cl::Device> devices;

		platforms[i].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &devices);

		sstream << "\n   Found " << devices.size() << " device(s):" << std::endl;

		for (unsigned int j = 0; j < devices.size(); j++)
		{
			sstream << "\n      Device " << j << ", " << devices[j].getInfo<CL_DEVICE_NAME>() << ", version: " << devices[j].getInfo<CL_DEVICE_VERSION>();

			sstream << ", vendor: " << devices[j].getInfo<CL_DEVICE_VENDOR>();
			cl_device_type device_type = devices[j].getInfo<CL_DEVICE_TYPE>();
			sstream << ", type: ";
			if (device_type & CL_DEVICE_TYPE_DEFAULT)
				sstream << "DEFAULT ";
			if (device_type & CL_DEVICE_TYPE_CPU)
				sstream << "CPU ";
			if (device_type & CL_DEVICE_TYPE_GPU)
				sstream << "GPU ";
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
				sstream << "ACCELERATOR ";
			sstream << ", compute units: " << devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			sstream << ", clock freq [MHz]: " << devices[j].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
			sstream << ", max memory size [B]: " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			sstream << ", max allocatable memory [B]: " << devices[j].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

			sstream << std::endl;
		}
	}
	sstream << "----------------------------------------------------------------" << std::endl;

	return sstream.str();
}

cl::Context GetContext(int platform_id, int device_id) {
	std::vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		std::vector<cl::Device> devices;
		platforms[i].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &devices);

		for (unsigned int j = 0; j < devices.size(); j++)
		{
			if ((i == platform_id) && (j == device_id))
				return cl::Context({ devices[j] });
		}
	}

	return cl::Context();
}

enum ProfilingResolution {
	PROF_NS = 1,
	PROF_US = 1000,
	PROF_MS = 1000000,
	PROF_S = 1000000000,
	PROF_NULL = -1
};

unsigned long* GetFullProfilingInfoData(const cl::Event& evnt, ProfilingResolution resolution)
{
	return new unsigned long[4] {
		(unsigned long)((evnt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / resolution),
		(unsigned long)((evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) / resolution),
		(unsigned long)((evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / resolution),
		(unsigned long)((evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / resolution)
	};
}

std::string GetFullProfilingInfo(unsigned long* profiled_info)
{
	std::stringstream sstream;

	sstream << "\n\tQueued " << profiled_info[0];
	sstream << "\n\tSubmitted " << profiled_info[1];
	sstream << "\n\tExecuted " << profiled_info[2];
	sstream << "\n\tTotal " << profiled_info[3];

	return sstream.str();
}

const char* GetResolutionString(ProfilingResolution resolution)
{
	switch (resolution)
	{
		case PROF_NS: return "[ns]";
		case PROF_US: return "[us]";
		case PROF_MS: return "[ms]";
		case PROF_S: return "[s]";
		default: break;
	}
}

#endif