// ------------------------------------------------------------------------------------------------------//
// ------------------------------------------ WORKSHOP KERNELS ------------------------------------------//
// ------------------------------------------------------------------------------------------------------//

// ------------------------------------------ REDUCTION ------------------------------------------ //
// W_REDUCTION
__kernel void reduce_add(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		//if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add(&B[0], scratch[lid]);
}

// ------------------------------------------ BITONIC SORTS ------------------------------------------ //
// W_CMPXCHG
void cmpxchg(__global int* A, __global int* B, bool dir)
{
	if ((!dir && *A > *B) || (dir && *A < *B))
	{
		int t = *A;
		*A = *B;
		*B = t;
	}
}

// W_BITONIC_MERGE
void bitonic_merge(int id, __global int* A, int N, bool dir)
{
	for (int i = N/2; i > 0; i /= 2)
	{
		if ((id % (i * 2)) < i)
			cmpxchg(&A[id], &A[id + i], dir);
			
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// W_BitonicSort
__kernel void sort_bitonic(__global int* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	for (int i = 1; i < N/2; i *= 2)
	{
		if (id % (i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge(id, A, i*2, true);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	bitonic_merge(id, A, N, false);
}

// ------------------------------------------ MISC KERNELS ------------------------------------------ //
// W_SIMPLE_HISTOGRAM
__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	int bin_index = A[id];

	atomic_inc(&H[bin_index]);
}

// W_SCAN_ADD
__kernel void scan_add(__global const int* A, __global int* B, __local int* scratch_1, __local int* scratch_2)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	__local int *scratch_3;

	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	B[id] = scratch_1[lid];
}

// W_BLOCK_SUM
__kernel void block_sum(__global const int* A, __global int* B, int local_size)
{
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

// W_SCAN_ADD_ATOMIC
__kernel void scan_add_atomic(__global int* A, __global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

// W_SCAN_ADD_ADJUST
__kernel void scan_add_adjust(__global int* A, __global const int* B)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
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

__kernel void bitonic_local(__global const int* in, __global int* out, __local int* cache)
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
__kernel void reduce_min(__global const int* in, __global int* out, __local int* scratch)
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
__kernel void reduce_min_global(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++)
	{
		if (in[id+i] < out[id])
			out[id] = in[id+i];
	}

	atomic_min(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max(__global const int* in, __global int* out, __local int* scratch)
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
__kernel void reduce_max_global(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++)
	{
		if (in[id+i] > out[id])
			out[id] = in[id+i];
	}

	atomic_max(&out[0], out[id]);
}