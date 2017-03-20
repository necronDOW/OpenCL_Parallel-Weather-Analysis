#ifndef analytics_h
#define analytics_h

#include <chrono>

namespace timer
{
	std::chrono::time_point<std::chrono::steady_clock> start;

	void Start() { start = std::chrono::high_resolution_clock::now(); }
	void Stop() { start = std::chrono::time_point<std::chrono::steady_clock>(); }
	void Reset() { Start(); }
	int QuerySeconds() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count(); }
	int QueryMilliseconds() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count(); }
	int QueryMicroseconds() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count(); }
	int QueryNanoseconds() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count(); }
}

#endif