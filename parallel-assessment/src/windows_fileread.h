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

/* This function is called when parsing data from a character array, and takes in index values to provide a head start when checking
   a single one dimensions character array. By doing this, the windows file reading implementation can remain the more optimal solution
   and N number of floats can be parsed from one sincle dimensional char*. */
fp_type ParseDouble(const char*& data, unsigned int len, unsigned int& index, int max_len)
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
	/* These two implementations of QueryLineCount operate in different ways, the second of the two is the quickest by far as it does not rely
	   upon slow ifstream file reading to operate and calculate the line count. Instead, the second implementation assumes an array has been
	   provided (read from ReadOptimal?) and counts each '\n' char to get the line count. */
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

	/* This file reading implementation makes use of core windows API functionality to achieve impressive speeds. The goal here was to reduce
	   the time taken to complete what is likely the largest bottleneck of the program as a whole, which is the sequential file reading.
	   Investigation found that standard ifstream was reading in around ~35s whereas a slightly faster fscanf (seen one function down) solution 
	   read at ~9s (both debug timings). This reading algorithm managed to read the entire 1.8 million lines from the text file in ~20ms, a 
	   massive improvement. */
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

	fp_type* Read_fscanf(const char* dir, unsigned int size)
	{
		fp_type* values = new fp_type[size];

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
			file << data << "\n" << TimeStamp() << "\n\n";
			file.close();
		}
	}

	/* This function partially relies upon naive assumptions of string length for a given numerical value. However, by doing this
	   it is possible to parse the input array into another array of floats in around ~500ms in debug with massive performance
	   boosts when ran in Release. I do not believe this is the fastest option, however it has enough safety checks in place to
	   ensure the correct values are parsed.*/
	fp_type* ParseLines(const char*& data, unsigned int len, char delimiter, unsigned char column_index, int size)
	{
		unsigned char current_column = 0;
		unsigned int index = 0;
		fp_type* out_data = new fp_type[size];

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