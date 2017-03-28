#ifndef windowsfileread_h
#define windowsfileread_h

#include <iostream>
#include <fstream>
#include <windows.h>
#include <sys/stat.h>
#include <ctime>

#include "paths.h"

unsigned int g_BytesTransferred = 0;

VOID CALLBACK FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped)
{
	//std::cout << "Number of bytes=" << dwNumberOfBytesTransfered << std::endl;
	g_BytesTransferred = dwNumberOfBytesTransfered;
}

unsigned int ComputeBytes(const char* dir)
{
	struct stat fileStatus;
	stat(dir, &fileStatus);

	return fileStatus.st_size - 1;
}

double ParseDouble(const char*& data, unsigned int len, unsigned int& index, int max_len)
{
	char* buffer = new char[max_len];
	for (int i = 0; i < max_len; i++)
	{
		if (data[index] == '\n')
			break;
		else buffer[i] = data[index++];
	}

	try { return atof(buffer); }
	catch (...) { return NULL; }
}

std::string TimeStamp()
{
	time_t t = time(0);
	struct tm* now = localtime(&t);

	std::stringstream ss;
	ss << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec
		<< " " << now->tm_mday << "/" << (now->tm_mon+1) << "/" << (now->tm_year+1900);
	return ss.str();
}

namespace winstr
{
	size_t QueryLineCount(const char* dir)
	{
		char c; size_t size = 0;
		std::ifstream file(dir);

		while (file.get(c))
		{
			if (c == '\n')
				size++;
		}

		return size;
	}
	size_t QueryLineCount(const char*& arr, int len)
	{
		size_t size = 0;

		for (int i = 0; i < len; i++)
		{
			if (arr[i] == '\n')
				++size;
		}

		return ++size;
	}

	char* ReadOptimal(const char* dir, unsigned int& len)
	{
		std::cout << "Reading (dir='" << dir << "') ..." << std::endl;

		HANDLE hFile = CreateFile(dir, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
		if (hFile == INVALID_HANDLE_VALUE)
		{
			std::cout << "Unable to open file '" << dir << "'." << std::endl;
			return nullptr;
		}

		unsigned int  dwBytesRead = 0;
		unsigned int bufferSize = ComputeBytes(dir);
		char* ReadBuffer = new char[bufferSize];
		OVERLAPPED ol = { 0 };

		if (!ReadFileEx(hFile, ReadBuffer, bufferSize, &ol, FileIOCompletionRoutine))
		{
			std::cout << "Unable to read from file." << std::endl;
			CloseHandle(hFile);

			len = 0;
			return nullptr;
		}

		SleepEx(5000, TRUE);
		dwBytesRead = g_BytesTransferred;

		if (dwBytesRead > 0 && dwBytesRead <= bufferSize)
			ReadBuffer[dwBytesRead] = '\0';

		CloseHandle(hFile);

		len = dwBytesRead;
		return ReadBuffer;
	}

	double* Read_fscanf(const char* dir, unsigned int size)
	{
		double* values = new double[size];

		FILE* stream = fopen(dir, "r");
		fseek(stream, 0L, SEEK_SET);

		for (unsigned int i = 0; i < size; i++)
			fscanf(stream, "%*s %*lf %*lf %*lf %*lf %d", &values[i]);

		fclose(stream);
		return values;
	}

	void Write(const char* data)
	{
		std::string dir = base_path + "logs/profiler_log.txt";
		std::ofstream file(dir, std::ios::app);

		time_t t = time(0);
		struct tm* now = localtime(&t);

		if (file.is_open())
		{
			file << data << "    " << TimeStamp() << "\n";
			file.close();
		}
	}

	double* ParseLines(const char*& data, unsigned int len, char delimiter, unsigned char column_index, int size)
	{
		unsigned char current_column = 0;
		unsigned int index = 0;
		double* out_data = new double[size];

		for (unsigned int i = 0; i < len; i++)
		{
			if (data[i] == delimiter || (data[i] == '\r' || data[i] == '\n'))
				current_column++;

			if (current_column == column_index)
			{
				if (index == size)
					break;

				out_data[index++] = ParseDouble(data, len, ++i, 6);
				current_column = 0;
			}
		}

		return out_data;
	}
};

#endif