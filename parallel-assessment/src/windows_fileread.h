#ifndef windowsfileread_h
#define windowsfileread_h

#include <iostream>
#include <fstream>
#include <windows.h>
#include <sys/stat.h>

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

double parse_double(const char*& data, unsigned int len, unsigned int& index, int max_len)
{
	char* buffer = new char[max_len];
	for (int i = 0; i < max_len; i++)
	{
		if (data[index] == '\n')
			break;

		buffer[i] = data[index++];
	}

	try { return atof(buffer); }
	catch (...) { return NULL; }
}

namespace winstr
{
	size_t query_line_count(const char* dir)
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
	size_t query_line_count(const char*& arr, int len)
	{
		size_t size = 0;

		for (int i = 0; i < len; i++)
		{
			if (arr[i] == '\n')
				++size;
		}

		return ++size;
	}

	char* read_optimal(const char* dir, unsigned int& len)
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

	double* read_fscanf(const char* dir, unsigned int size)
	{
		double* values = new double[size];

		FILE* stream = fopen(dir, "r");
		fseek(stream, 0L, SEEK_SET);

		for (unsigned int i = 0; i < size; i++)
			fscanf(stream, "%*s %*lf %*lf %*lf %*lf %d", &values[i]);

		fclose(stream);
		return values;
	}

	double* parse_lines(const char*& data, unsigned int len, char delimiter, unsigned char column_index, int size)
	{
		//std::cout << "Parsing (size=" << size << ") ..." << std::endl;

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

				out_data[index++] = parse_double(data, len, ++i, 5);
				current_column = 0;
			}
		}

		return out_data;
	}
};

#endif