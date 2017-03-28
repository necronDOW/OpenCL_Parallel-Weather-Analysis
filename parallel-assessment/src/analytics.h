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

	unsigned long Query(ProfilingResolution resolution)
	{
		unsigned long new_query = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
		since_last = new_query - last_query;
		last_query = new_query;

		return new_query / resolution;
	}
	unsigned long QuerySinceLast(ProfilingResolution resolution)
	{
		Query(PROF_NS);
		return since_last / resolution;
	}
}

#endif