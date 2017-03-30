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

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
			scratch[lid] += scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}

// ------------------------------------------ TWO-STAGE_REDUCTION ------------------------------------------ //
// REDUCE_MIN_LOCAL
__kernel void reduce_min_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int lid = get_local_id(0);
	scratch[lid] = in[get_global_id(0)];

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			int other = scratch[lid+i];
			int mine = scratch[lid];
			scratch[lid] = (mine < other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_min(&out[0], scratch[lid]);
}

// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_INT(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			int other = in[id+i];
			int mine = out[id];
			out[id] = (mine < other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	atomic_min(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int lid = get_local_id(0);
	scratch[lid] = in[get_global_id(0)];

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			int other = scratch[lid+i];
			int mine = scratch[lid];
			scratch[lid] = (mine > other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_max(&out[0], scratch[lid]);
}

// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_INT(__global const int* in, __global int* out)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			int other = in[id+i];
			int mine = out[id];
			out[id] = (mine > other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	atomic_max(&out[0], out[id]);
}

// ------------------------------------------ VARIANCE ------------------------------------------ //
// SUM_SQR_DIFF
__kernel void sum_sqr_diff_INT(__global const int* in, __global int* out, __local int* scratch, int mean)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	int diff = (in[id] - mean);
	scratch[lid] = (diff * diff) / 10;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}

// ------------------------------------------ SORT ------------------------------------------ //
// BITONIC_SORT
__kernel void bitonic_local_INT(__global const int* in, __global int* out, __local int* scratch, int merge)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	int max_group = (get_global_size(0) / N) - 1;
	int offset_id = id + ((N/2) * merge);

	if (merge && gid == 0)
	{
		out[id] = in[id];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	scratch[lid] = in[offset_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int l=1; l<N; l<<=1)
	{
		bool direction = ((lid & (l<<1)) != 0);

		for (int inc=l; inc>0; inc>>=1)
		{
			int j = lid ^ inc;
			int i_data = scratch[lid];
			int j_data = scratch[j];

			bool smaller = (j_data < i_data) || ( j_data == i_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			barrier(CLK_LOCAL_MEM_FENCE);

			scratch[lid] = (swap) ? j_data : i_data;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[offset_id] = scratch[lid];
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (merge && gid == max_group)
		out[offset_id] = in[offset_id];
}

// #################################################################################################### //
// ########################################## DOUBLE KERNELS ########################################## //
// #################################################################################################### //

typedef float fp_type;

// ----------------------------------------------------------------------------------------------------//
// ------------------------------------------ ATOMIC KERNELS ------------------------------------------//
// ----------------------------------------------------------------------------------------------------//

void atomic_add_f(volatile __global fp_type* dst, const fp_type delta)
{
	union { fp_type f; uint i; } oldVal;
	union { fp_type f; uint i; } newVal;
	do
	{ 
		oldVal.f = *dst; 
		newVal.f = oldVal.f + delta; 
	}
	while (atom_cmpxchg((volatile __global uint*)dst, oldVal.i, newVal.i) != oldVal.i);
}

void atomic_max_f(volatile __global fp_type* a, const fp_type b)
{
	union { fp_type f; uint i; } oldVal;
	union { fp_type f; uint i; } newVal;
	do
	{ 
		oldVal.f = *a; 
		newVal.f = max(oldVal.f, b); 
	}
	while (atom_cmpxchg((volatile __global uint*)a, oldVal.i, newVal.i) != oldVal.i);
}

void atomic_min_f(volatile __global fp_type* a, const fp_type b)
{
	union { fp_type f; uint i; } oldVal;
	union { fp_type f; uint i; } newVal;
	do
	{ 
		oldVal.f = *a; 
		newVal.f = min(oldVal.f, b); 
	}
	while (atom_cmpxchg((volatile __global uint*)a, oldVal.i, newVal.i) != oldVal.i);
}

// ------------------------------------------------------------------------------------------------------//
// ------------------------------------------ WORKSHOP KERNELS ------------------------------------------//
// ------------------------------------------------------------------------------------------------------//

// ------------------------------------------ REDUCTION ------------------------------------------ //
// W_REDUCTION
__kernel void reduce_sum_FP(__global const fp_type* in, __global fp_type* out, __local fp_type* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
			scratch[lid] += scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add_f(&out[0], scratch[lid]);
}

// ---------------------------------------------------------------------------------------------------------//
// ------------------------------------------ ALTERNATIVE KERNELS ------------------------------------------//
// ---------------------------------------------------------------------------------------------------------//

// ------------------------------------------ TWO-STAGE_REDUCTION ------------------------------------------ //
// REDUCE_MIN_LOCAL
__kernel void reduce_min_FP(__global const fp_type* in, __global fp_type* out, __local fp_type* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			fp_type other = scratch[lid+i];
			fp_type mine = scratch[lid];
			scratch[lid] = (mine < other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_min_f(&out[0], scratch[lid]);
}

// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_FP(__global const fp_type* in, __global fp_type* out)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			fp_type other = in[id+i];
			fp_type mine = out[id];
			out[id] = (mine < other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	atomic_min_f(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max_FP(__global const fp_type* in, __global fp_type* out, __local fp_type* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			fp_type other = scratch[lid+i];
			fp_type mine = scratch[lid];
			scratch[lid] = (mine > other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_max_f(&out[0], scratch[lid]);
}

// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_FP(__global const fp_type* in, __global fp_type* out)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	out[id] = in[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			fp_type other = in[id+i];
			fp_type mine = out[id];
			out[id] = (mine > other) ? mine : other;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	atomic_max_f(&out[0], out[id]);
}

// ------------------------------------------ VARIANCE ------------------------------------------ //
// SUM_SQR_DIFF
__kernel void sum_sqr_diff_FP(__global const fp_type* in, __global fp_type* out, __local fp_type* scratch, fp_type mean)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	fp_type diff = (in[id] - mean);
	scratch[lid] = (diff * diff);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
		atomic_add_f(&out[0], scratch[lid]);
}

// ------------------------------------------ SORT ------------------------------------------ //
// BITONIC_SORT
__kernel void bitonic_local_FP(__global const fp_type* in, __global fp_type* out, __local fp_type* scratch, int merge)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	int max_group = (get_global_size(0) / N) - 1;
	int offset_id = id + ((N/2) * merge);

	if (merge && gid == 0)
	{
		out[id] = in[id];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	scratch[lid] = in[offset_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int l=1; l<N; l<<=1)
	{
		bool direction = ((lid & (l<<1)) != 0);

		for (int inc=l; inc>0; inc>>=1)
		{
			int j = lid ^ inc;
			fp_type i_data = scratch[lid];
			fp_type j_data = scratch[j];

			bool smaller = (j_data < i_data) || ( j_data == i_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			barrier(CLK_LOCAL_MEM_FENCE);

			scratch[lid] = (swap) ? j_data : i_data;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[offset_id] = scratch[lid];
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (merge && gid == max_group)
		out[offset_id] = in[offset_id];
}