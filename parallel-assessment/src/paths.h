#ifndef paths_h
#define paths_h

#include <windows.h>
#include <iostream>

// Typedef for the floating point type to be used, this can either be double or float in the current state of the program.
typedef float fp_type;

std::string base_path = "./";
std::string src_path = "./src/";
std::string kernel_path = "./src/kernels/";
std::string data_path = "./data/";

void InitPaths()
{
	// This function ensures that the correct paths are gathered whether running from Visual Studio or simply from command line.
	if (!IsDebuggerPresent())
	{
		base_path = "../../parallel-assessment/";
		src_path = "../../parallel-assessment/src/";
		kernel_path = "../../parallel-assessment/src/kernels/";
		data_path = "../../parallel-assessment/data/";
	}
}

#endif