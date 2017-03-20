__kernel void reduce_add_1(__global const int* A, __global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];
	barrier(CLK_GLOBAL_MEM_FENCE);
	 
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

__kernel void reduce_add_2(__global const int* A, __global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void reduce_add_3(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[id] = scratch[lid];
}

__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
	{
		atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	int bin_index = A[id];

	atomic_inc(&H[bin_index]);
}

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

__kernel void block_sum(__global const int* A, __global int* B, int local_size)
{
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

__kernel void scan_add_atomic(__global int* A, __global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

__kernel void scan_add_adjust(__global int* A, __global const int* B)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}
