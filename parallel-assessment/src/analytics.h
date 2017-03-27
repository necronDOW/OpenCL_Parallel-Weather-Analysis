#ifndef analytics_h
#define analytics_h

#include <chrono>

namespace analytics
{
	const char* BuildInfo()
	{
		#ifdef _DEBUG
			return "Build Mode: Debug\n";
		#else 
			return "Build Mode: Release\n";
		#endif
	}
}

namespace timer
{
	std::chrono::time_point<std::chrono::steady_clock> start;
	unsigned long since_last = 0;
	unsigned long last_query = 0;

	void Start() { start = std::chrono::high_resolution_clock::now(); }
	void Stop() { start = std::chrono::time_point<std::chrono::steady_clock>(); }
	void Reset() { Start(); }

	unsigned long QueryNanoseconds()
	{
		unsigned long new_query = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
		since_last = new_query - last_query;
		last_query = new_query;

		return new_query;
	}
	unsigned long QueryNanosecondsSinceLast()
	{
		QueryNanoseconds();
		return since_last;
	}

	unsigned long QueryMicroseconds() { return QueryNanoseconds() / 1000; }
	unsigned long QueryMicrosecondsSinceLast() { return QueryNanosecondsSinceLast() / 1000; }

	unsigned long QueryMilliseconds() { return QueryNanoseconds() / 1000000; }
	unsigned long QueryMillisecondsSinceLast() { return QueryNanosecondsSinceLast() / 1000000; }

	unsigned long QuerySeconds() { return QueryNanoseconds() / 1000000000; }
	unsigned long QuerySecondsSinceLast() { return QueryNanosecondsSinceLast() / 1000000000; }
}

#endif