#ifndef windowsfileread_h
#define windowsfileread_h

#include <iostream>
#include <fstream>
#include <windows.h>
#include <sys/stat.h>

unsigned int g_BytesTransferred = 0;

VOID CALLBACK FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped)
{
	std::cout << "Number of bytes=" << dwNumberOfBytesTransfered << std::endl;
	g_BytesTransferred = dwNumberOfBytesTransfered;
}

unsigned int ComputeBytes(const char* dir)
{
	struct stat fileStatus;
	stat(dir, &fileStatus);

	return fileStatus.st_size - 1;
}

double parse_float(const char*& data, unsigned int len, char delimiter, unsigned int& index, int accuracy)
{
	short buffer_index = 0;
	char* buffer = new char[accuracy];

	for (index; index < len; index++)
	{
		if (data[index] == delimiter || (data[index] == '\r' || data[index] == '\n'))
			break;
		else buffer[buffer_index++] = data[index];
	}

	index++;
	try { return stof(buffer); }
	catch (...) { return NULL; }
}

namespace winstr
{
	char* read_optimal(const char* dir, unsigned int& len)
	{
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

	double* read_fscanf(const char* dir, unsigned int lineCount)
	{
		double* values = new double[lineCount];

		FILE* stream = fopen(dir, "r");
		fseek(stream, 0L, SEEK_SET);

		for (unsigned int i = 0; i < lineCount; i++)
			fscanf(stream, "%*s %*lf %*lf %*lf %*lf %d", &values[i]);

		fclose(stream);
		return values;
	}

	double* parse_lines(const char*& data, unsigned int len, char delimiter, unsigned char column_index, int size)
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

				out_data[index++] = parse_float(data, len, delimiter, ++i, 4);
				current_column = 0;
			}
		}

		return out_data;
	}
};

#endif