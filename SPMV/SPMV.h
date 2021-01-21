typedef struct {
	float *h_val, *h_vec, *refOut, *h_out;
	int *h_cols, *h_rowDelimiters;

	float *d_val, *d_out, *d_vec;
	int *d_cols, *d_rowDelimiters;
	
	int numNonZeroes, numRows;
	int nItems;
	int gridDimX;
	int *zc_slc;
} t_SPMV_params;

int launch_preemp_SPMVcsr(void *arg);
int launch_orig_SPMVcsr(void *arg);
int launch_slc_SPMVcsr(void *arg);
int SPMVcsr_start_kernel(void *arg);
int SPMVcsr_end_kernel(void *arg);
int SPMVcsr_start_transfers(void *arg);
int SPMVcsr_start_mallocs(void *arg);

__global__ void
original_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out);