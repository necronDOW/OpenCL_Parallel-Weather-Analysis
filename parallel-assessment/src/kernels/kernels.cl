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
__kernel void bitonic_local_INT(__global int* in, __global int* out, __local int* scratch, int merge)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	int max_group = (get_global_size(0) / N) - 1;
	int offset = (N/2) * merge;

	if (merge && gid == 0)
	{
		out[id] = in[id];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	scratch[lid] = in[id+offset];
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

	out[id+offset] = scratch[lid];
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (merge && gid == max_group)
		out[id+offset] = in[id+offset];
}

// QUICK_SORT
__kernel void quick_sort_INT(__global int* data, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int end = get_local_size(0)-1;

	scratch[lid] = data[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	printf("%d\n", scratch[lid]);

	int stack[32];
	int top = -1;

	stack[++top] = lid;
	stack[++top] = end;

	while (top >= 0)
	{
		end = stack[top--];
		lid = stack[top--];

		// partition
		int x = scratch[end];
		int i = (lid-1);

		for (int j = lid; j <= end-1; j++)
		{
			if (scratch[j] <= x)
			{
				i++;

				int t = scratch[i];
				scratch[i] = scratch[j];
				scratch[j] = t;
			}
		}

		int t = scratch[i+1];
		scratch[i+1] = scratch[end];
		scratch[end] = t;

		int p = i+1;

		if (p-1 > lid)
		{
			stack[++top] = lid;
			stack[++top] = p-1;
		}

		if (p+1 < end)
		{
			stack[++top] = p+1;
			stack[++top] = end;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	data[id] = scratch[lid];
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
__kernel void reduce_sum_DOUBLE(__global const double* in, __global double* out, __local double* scratch)
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
__kernel void reduce_min_DOUBLE(__global const double* in, __global double* out, __local double* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

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
		atomic_min_f(&out[0], scratch[lid]);
}

// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_DOUBLE(__global const double* in, __global double* out)
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

	atomic_min_f(&out[0], out[id]);
}

// REDUCE_MAX_LOCAL
__kernel void reduce_max_DOUBLE(__global const double* in, __global double* out, __local double* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	scratch[lid] = in[id];

	barrier(CLK_LOCAL_MEM_FENCE);

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
		atomic_max_f(&out[0], scratch[lid]);
}

// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_DOUBLE(__global const double* in, __global double* out)
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

	atomic_max_f(&out[0], out[id]);
}

// ------------------------------------------ VARIANCE ------------------------------------------ //
// SUM_SQR_DIFF
__kernel void sum_sqr_diff_DOUBLE(__global const double* in, __global double* out, __local double* scratch, double mean)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	double diff = (in[id] - mean);
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