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
	long long since_last = 0;
	long long last_query = 0;

	long long Query(ProfilingResolution resolution)
	{
		long long new_query = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
		since_last = new_query - last_query;
		last_query = new_query;

		return new_query / resolution;
	}
	long long QuerySinceLast(ProfilingResolution resolution)
	{
		Query(PROF_NS);
		return since_last / resolution;
	}

	void Start() { start = std::chrono::high_resolution_clock::now(); }
	long long Stop(ProfilingResolution resolution = PROF_NULL)
	{
		long long query = Query(resolution);

		start = std::chrono::time_point<std::chrono::steady_clock>();
		return query;
	}
	void Reset() { Start(); }
}

#endif