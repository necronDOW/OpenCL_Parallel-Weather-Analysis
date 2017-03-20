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

namespace winfr
{
	char* Read(const char* dir)
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
			return nullptr;
		}

		SleepEx(5000, TRUE);
		dwBytesRead = g_BytesTransferred;

		if (dwBytesRead > 0 && dwBytesRead <= bufferSize)
			ReadBuffer[dwBytesRead] = '\0';

		CloseHandle(hFile);

		return ReadBuffer;
	}
};

#endif