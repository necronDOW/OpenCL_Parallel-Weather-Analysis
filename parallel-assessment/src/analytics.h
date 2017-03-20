#ifndef analytics_h
#define analytics_h

#include <chrono>

using namespace std::chrono;

namespace timer
{
	time_point<steady_clock> start;

	void Start() { start = high_resolution_clock::now(); }
	void Stop() { start = time_point<steady_clock>(); }
	void Reset() { Start(); }
	int QuerySeconds() { return duration_cast<seconds>(high_resolution_clock::now() - start).count(); }
	int QueryMilliseconds() { return duration_cast<milliseconds>(high_resolution_clock::now() - start).count(); }
	int QueryMicroseconds() { return duration_cast<microseconds>(high_resolution_clock::now() - start).count(); }
	int QueryNanoseconds() { return duration_cast<nanoseconds>(high_resolution_clock::now() - start).count(); }
}

#endif