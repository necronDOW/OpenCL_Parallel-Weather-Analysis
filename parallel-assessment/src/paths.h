#ifndef paths_h
#define paths_h

#include <windows.h>
#include <iostream>

std::string base_path = "./";
std::string src_path = "./src/";
std::string kernel_path = "./src/kernels/";
std::string data_path = "./data/";

void InitPaths()
{
	if (!IsDebuggerPresent())
	{
		base_path = "../../parallel-assessment/";
		src_path = "../../parallel-assessment/src/";
		kernel_path = "../../parallel-assessment/src/kernels/";
		data_path = "../../parallel-assessment/data/";
	}
}

#endif