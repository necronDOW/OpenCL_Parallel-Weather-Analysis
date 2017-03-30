#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

// ##################################################################################################### //
// ########################################## INTEGER KERNELS ########################################## //
// ##################################################################################################### //

// The integer kernels operate on the original temperature values multiplied by 10.0 to account for the  //
// single decimal place. These operate far faster than the float operations later within this kernel but //
// display minor discrepancies in results when compared to the floating point operations.                //



// REDUCE_SUM
__kernel void reduce_sum_INT(__global const int* in, __global int* out, __local int* scratch)
{
	// Get the global ID for writing to global memory buffers (in & out), and get the local id to point to the correct location in the scratch.
	int id = get_global_id(0);
	int lid = get_local_id(0);

	/* Write N elements to the scratch from the in buffer, where N is equal to the work group siz. This allows for 
	   working on local memory and minimal global calls. */
	scratch[lid] = in[id];

	// Wait for all threads to finish/sync local memory operations up to this point.
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
			scratch[lid] += scratch[lid+i];

		// Wait for all threads to finish/sync local memory operations up to this point.
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* If the local id is equal to 0, perform an atomic_add operation to sum the first element of each scratch into
	   the first element of the output buffer [0] */
	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}



// REDUCE_MIN_LOCAL
__kernel void reduce_min_INT(__global const int* in, __global int* out, __local int* scratch)
{
	// Get the local id to point to the correct location in the scratch.
	int lid = get_local_id(0);

	/* Write N elements to the scratch from the in buffer, where N is equal to the work group siz. This allows for 
	   working on local memory and minimal global calls. */
	scratch[lid] = in[get_global_id(0)];

	// Wait for all threads to finish/sync local memory operations up to this point.
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
		{
			// Swap elements at index lid and lid+1 in the local scratch;
			int other = scratch[lid+i];
			int mine = scratch[lid];
			scratch[lid] = (mine < other) ? mine : other;
		}

		// Wait for all threads to finish/sync local memory operations up to this point.
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* If the local id is equal to 0, perform an atomic_min operation to sum the first element of each scratch into
	   the first element of the output buffer [0] */
	if (!lid)
		atomic_min(&out[0], scratch[lid]);
}



// REDUCE_MIN_GLOBAL
__kernel void reduce_min_global_INT(__global const int* in, __global int* out)
{
	/* Get the global ID for writing to global memory buffers (in & out), and get the local id to ensure that 
	   operations take place within the work group. */
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// Buffer N elements, where N is the work group size, from the input to the output buffer at index id.
	out[id] = in[id];

	// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
	barrier(CLK_GLOBAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
		{
			// Swap elements at index id+1 on the in buffer, with element at index id on the out buffer.
			int other = in[id+i];
			int mine = out[id];
			out[id] = (mine < other) ? mine : other;
		}

		// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Perform an atomic_min operation to sum out[id] to out[0] when the local id is equal to 0 (first element of work group).
	atomic_min(&out[0], out[id]);
}



// REDUCE_MAX_LOCAL
__kernel void reduce_max_INT(__global const int* in, __global int* out, __local int* scratch)
{
	int lid = get_local_id(0);

	/* Write N elements to the scratch from the in buffer, where N is equal to the work group siz. This allows for 
	   working on local memory and minimal global calls. */
	scratch[lid] = in[get_global_id(0)];

	// Wait for all threads to finish/sync local memory operations up to this point.
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
		{
			// Swap elements at index lid and lid+1 in the local scratch;
			int other = scratch[lid+i];
			int mine = scratch[lid];
			scratch[lid] = (mine > other) ? mine : other;
		}

		// Wait for all threads to finish/sync local memory operations up to this point.
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* If the local id is equal to 0, perform an atomic_max operation to sum the first element of each scratch into
	   the first element of the output buffer [0] */
	if (!lid)
		atomic_max(&out[0], scratch[lid]);
}



// REDUCE_MAX_GLOBAL
__kernel void reduce_max_global_INT(__global const int* in, __global int* out)
{
	/* Get the global ID for writing to global memory buffers (in & out), and get the local id to ensure that 
	   operations take place within the work group. */
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// Buffer N elements, where N is the work group size, from the input to the output buffer at index id.
	out[id] = in[id];

	// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
	barrier(CLK_GLOBAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
		{
			// Swap elements at index id+1 on the in buffer, with element at index id on the out buffer.
			int other = in[id+i];
			int mine = out[id];
			out[id] = (mine > other) ? mine : other;
		}

		// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Perform an atomic_min operation to sum out[id] to out[0] when the local id is equal to 0 (first element of work group).
	atomic_max(&out[0], out[id]);
}



// SUM_SQR_DIFF
__kernel void sum_sqr_diff_INT(__global const int* in, __global int* out, __local int* scratch, int mean)
{
	// Get the global ID for writing to global memory buffers (in & out), and get the local id to point to the correct location in the scratch.
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// Calculate the input[id] - mean and copy its squared value to the scratch at index lid.
	int diff = (in[id] - mean);
	scratch[lid] = (diff * diff) / 10;

	// Wait for all threads to finish/sync local memory operations up to this point.
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		// Wait for all threads to finish/sync local memory operations up to this point.
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* If the local id is equal to 0, perform an atomic_add operation to sum the first element of each scratch into
	   the first element of the output buffer [0] */
	if (!lid)
		atomic_add(&out[0], scratch[lid]);
}



// BITONIC_SORT
__kernel void bitonic_local_INT(__global const int* in, __global int* out, __local int* scratch, int merge)
{
	// Get the global ID for writing to global memory buffers (in & out), and get the local id to point to the correct location in the scratch.
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// Get the work group id. This helps when catching the 'stray' values when shifting the scratch offset.
	int gid = get_group_id(0);

	// Get the local size of the work group, which should be coherent across all groups.
	int N = get_local_size(0);

	// Calculate the maximum group index possible by dividing the global size by the work group size and subtracting 1.
	int max_group = (get_global_size(0) / N) - 1;

	/* Representation of the offset id based on the value of merge. This is crucial when sorting between work groups in local space.
	   An example of how it works is: when N = 1024, offset_id alternates between 0 and 512. */
	int offset_id = id + ((N/2) * merge);

	// If a merge is taking place between groups and this group is the first.
	if (merge && gid == 0)
	{
		// Buffer all of the elements from this group into the output.
		out[id] = in[id];

		// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	/* Write N elements to the scratch from the in buffer, where N is equal to the work group siz. This allows for 
	   working on local memory and minimal global calls. */
	scratch[lid] = in[offset_id];

	// Wait for all threads to finish/sync local memory operations up to this point.
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i*=2 operations. */
	for (int l=1; l<N; l<<=1)
	{
		// Set the direction bool for this particular run of bitonic sort. This equates to 0 or 1 which helps for mathematical formulas.
		bool direction = ((lid & (l<<1)) != 0);

		/* Access local memory in a commutative manner, bitshifting i after each iteration to provide optimal execution time
	   over typical i/=2 operations. */
		for (int inc=l; inc>0; inc>>=1)
		{
			// Gather the two data points to compare and store in i_data and j_data.
			int j = lid ^ inc;
			int i_data = scratch[lid];
			int j_data = scratch[j];

			/* Check if i_data < j_data and perform bitwise operations on the result combined with the direction, as well as 
				whether j is within the work group, to determine if a swap should take place. */
			bool smaller = (j_data < i_data) || ( j_data == i_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			// Wait for all threads to finish/sync local memory operations up to this point.
			barrier(CLK_LOCAL_MEM_FENCE);

			// If a swap should happen, place the smallest value within the scratch buffer, at index lid.
			scratch[lid] = (swap) ? j_data : i_data;

			// Wait for all threads to finish/sync local memory operations up to this point.
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	// Copy the values from the scratch into the output buffer.
	out[offset_id] = scratch[lid];

	// Wait for all threads to finish/sync global memory operations up to this point, this is very inefficient!
	barrier(CLK_GLOBAL_MEM_FENCE);

	// If a merge is taking place between groups and this group is the last, copy the last N/2 values from the input to the output buffer.
	if (merge && gid == max_group)
		out[offset_id] = in[offset_id];
}

// #################################################################################################### //
// ########################################## DOUBLE KERNELS ########################################## //
// #################################################################################################### //

// The kernels seen below are identical to those above with subtle differences. Firstly, the kernels    //
// below operator on floating point numbers rather than integers and thus provide 100% accuracy when    //
// compared to the original data input. This results in the requirement for custom atomic operations    //
// which are seen below. Secondly, the calls to atomic are slightly different due to the customized     //
// functions.                                                                                           //

// ----------------------------------------------------------------------------------------------------//
// ------------------------------------------ ATOMIC KERNELS ------------------------------------------//
// ----------------------------------------------------------------------------------------------------//

typedef float fp_type;

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



// REDUCE_SUM
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

		barrier(CLK_GLOBAL_MEM_FENCE);
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

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	atomic_max_f(&out[0], out[id]);
}



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