#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

// ##################################################################################################### //
// ########################################## INTEGER KERNELS ########################################## //
// ##################################################################################################### //

// ------------------------------------------------------------------------------------------------------//
// ------------------------------------------ WORKSHOP KERNELS ------------------------------------------//
// ------------------------------------------------------------------------------------------------------//

// ------------------------------------------ REDUCTION ------------------------------------------ //
// W_REDUCTION
__kernel void reduce_sum_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		//if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}

// ------------------------------------------ BITONIC SORTS ------------------------------------------ //
// W_CMPXCHG
void cmpxchg_INT(__global int* A, __global int* B, bool dir)
{
	if ((!dir && *A > *B) || (dir && *A < *B))
	{
		int t = *A;
		*A = *B;
		*B = t;
	}
}

// W_BITONIC_MERGE
void bitonic_merge_INT(int id, __global int* data, int N, bool dir)
{
	for (int i = N/2; i > 0; i /= 2)
	{
		if ((id % (i * 2)) < i)
			cmpxchg_INT(&data[id], &data[id + i], dir);
			
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// W_BitonicSort
__kernel void sort_bitonic_INT(__global int* data)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	for (int i = 1; i < N/2; i *= 2)
	{
		if (id % (i*4) < i*2)
			bitonic_merge_INT(id, data, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge_INT(id, data, i*2, true);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	bitonic_merge_INT(id, data, N, false);
}

// ---------------------------------------------------------------------------------------------------------//
// ------------------------------------------ ALTERNATIVE KERNELS ------------------------------------------//
// ---------------------------------------------------------------------------------------------------------//

// ------------------------------------------ BITONIC SORT (http://www.bealto.com/gpu-sorting_parallel-bitonic-local.html) ------------------------------------------ //
#if CONFIG_USE_VALUE
	#define getKey(a) ((a).x)
	#define getValue(a) ((a).y)
	#define makeData(k,v) ((int)((k),(v)))
#else
	#define getKey(a) (a)
	#define getValue(a) (0)
	#define makeData(k,v) (k)
#endif

__kernel void bitonic_local_INT(__global const int* in, __global int* out, __local int* cache)
{
	int lid = get_local_id(0);
	int wg = get_local_size(0);

	int offset = get_group_id(0) * wg;
	in += offset; out += offset;

	cache[lid] = in[lid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int len = 1; len < wg; len <<= 1)
	{
		bool dir = ((lid & (len << 1)) != 0);

		for (int inc = len; inc > 0; inc >>= 1)
		{
			int j = lid ^ inc;
			int i_data = cache[lid];
			uint i_key = getKey(i_data);
			int j_data = cache[j];
			uint j_key = getKey(j_data);
			bool smaller = (j_key < i_key) || (j_key == i_key && j < lid);
			bool swap = smaller ^ (j < lid) ^ dir;
			barrier(CLK_LOCAL_MEM_FENCE);

			cache[lid] = (swap) ? j_data : i_data;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[lid] = cache[lid];
}

// ------------------------------------------ TWO-STAGE_REDUCTION ------------------------------------------ //
// REDUCE_MIN_LOCAL
__kernel void reduce_min_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[lid+i] < scratch[lid])
			scratch[lid] = in[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_min(&out[0], scratch[lid]);
}

// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_INT(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[id+i] < out[id])
			out[id] = in[id+i];
	}

	atomic_min(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[lid+i] > scratch[lid])
			scratch[lid] = in[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_max(&out[0], scratch[lid]);
}

// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_INT(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[id+i] > out[id])
			out[id] = in[id+i];
	}

	atomic_max(&out[0], out[id]);
}

// ------------------------------------------ VARIANCE ------------------------------------------ //
__kernel void sum_sqr_diff_INT(__global const int* in, __global int* out, __local int* scratch, int mean)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	int diff = (in[id] - mean);
	scratch[lid] = (diff * diff) / 10;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}

// #################################################################################################### //
// ########################################## DOUBLE KERNELS ########################################## //
// #################################################################################################### //

// ----------------------------------------------------------------------------------------------------//
// ------------------------------------------ ATOMIC KERNELS ------------------------------------------//
// ----------------------------------------------------------------------------------------------------//

void atomic_add_f(__global double* dst, double delta)
{
	union { double f; ulong ul; } old;
	union { double f; ulong ul; } new;
	do
	{ 
		old.f = *dst; 
		new.f = old.f + delta; 
	}
	while (atom_cmpxchg((volatile __global ulong*)dst, old.ul, new.ul) != old.ul);
}

void atomic_max_f(__global double* a, double b)
{
	union { double f; ulong ul; } old;
	union { double f; ulong ul; } new;
	do
	{ 
		old.f = *a; 
		new.f = max(old.f, b); 
	}
	while (atom_cmpxchg((volatile __global ulong*)a, old.ul, new.ul) != old.ul);
}

void atomic_min_f(__global double* a, double b)
{
	union { double f; ulong ul; } old;
	union { double f; ulong ul; } new;
	do
	{ 
		old.f = *a; 
		new.f = min(old.f, b); 
	}
	while (atom_cmpxchg((volatile __global ulong*)a, old.ul, new.ul) != old.ul);
}

// ------------------------------------------------------------------------------------------------------//
// ------------------------------------------ WORKSHOP KERNELS ------------------------------------------//
// ------------------------------------------------------------------------------------------------------//

// ------------------------------------------ REDUCTION ------------------------------------------ //
// W_REDUCTION
__kernel void reduce_sum_DOUBLE(__global volatile const double* in, __global volatile double* out, __local volatile double* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		//if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add_f(&out[0], scratch[lid]);
}

// ------------------------------------------ BITONIC SORTS ------------------------------------------ //
// W_CMPXCHG
void cmpxchg_DOUBLE(__global double* A, __global double* B, bool dir)
{
	if ((!dir && *A > *B) || (dir && *A < *B))
	{
		double t = *A;
		*A = *B;
		*B = t;
	}
}

// W_BITONIC_MERGE
void bitonic_merge_DOUBLE(int id, __global double* data, int N, bool dir)
{
	for (int i = N/2; i > 0; i /= 2)
	{
		if ((id % (i * 2)) < i)
			cmpxchg_DOUBLE(&data[id], &data[id + i], dir);
			
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// W_BitonicSort
__kernel void sort_bitonic_DOUBLE(__global double* data)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	for (int i = 1; i < N/2; i *= 2)
	{
		if (id % (i*4) < i*2)
			bitonic_merge_DOUBLE(id, data, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge_DOUBLE(id, data, i*2, true);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	bitonic_merge_DOUBLE(id, data, N, false);
}

// ---------------------------------------------------------------------------------------------------------//
// ------------------------------------------ ALTERNATIVE KERNELS ------------------------------------------//
// ---------------------------------------------------------------------------------------------------------//

// ------------------------------------------ BITONIC SORT (http://www.bealto.com/gpu-sorting_parallel-bitonic-local.html) ------------------------------------------ //
#if CONFIG_USE_VALUE
	#define getKey(a) ((a).x)
	#define getValue(a) ((a).y)
	#define makeData(k,v) ((int)((k),(v)))
#else
	#define getKey(a) (a)
	#define getValue(a) (0)
	#define makeData(k,v) (k)
#endif

__kernel void bitonic_local_DOUBLE(__global const double* in, __global double* out, __local int* cache)
{
	int lid = get_local_id(0);
	int wg = get_local_size(0);

	int offset = get_group_id(0) * wg;
	in += offset; out += offset;

	cache[lid] = in[lid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int len = 1; len < wg; len <<= 1)
	{
		bool dir = ((lid & (len << 1)) != 0);

		for (int inc = len; inc > 0; inc >>= 1)
		{
			int j = lid ^ inc;
			int i_data = cache[lid];
			uint i_key = getKey(i_data);
			int j_data = cache[j];
			uint j_key = getKey(j_data);
			bool smaller = (j_key < i_key) || (j_key == i_key && j < lid);
			bool swap = smaller ^ (j < lid) ^ dir;
			barrier(CLK_LOCAL_MEM_FENCE);

			cache[lid] = (swap) ? j_data : i_data;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[lid] = cache[lid];
}

// ------------------------------------------ TWO-STAGE_REDUCTION ------------------------------------------ //
// REDUCE_MIN_LOCAL
__kernel void reduce_min_DOUBLE(__global const double* in, __global double* out, __local double* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[lid+i] < scratch[lid])
			scratch[lid] = in[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_min_f(&out[0], scratch[lid]);
}

// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_DOUBLE(__global const double* in, __global double* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[id+i] < out[id])
			out[id] = in[id+i];
	}

	atomic_min_f(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max_DOUBLE(__global const double* in, __global double* out, __local double* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[lid+i] > scratch[lid])
			scratch[lid] = in[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_max_f(&out[0], scratch[lid]);
}

// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_DOUBLE(__global const double* in, __global double* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (in[id+i] > out[id])
			out[id] = in[id+i];
	}

	atomic_max_f(&out[0], out[id]);
}

// ------------------------------------------ VARIANCE ------------------------------------------ //
__kernel void sum_sqr_diff_DOUBLE(__global const double* in, __global double* out, __local double* scratch, double mean)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	double diff = (in[id] - mean);
	scratch[lid] = (diff * diff);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add_f(&out[0], scratch[lid]);
}