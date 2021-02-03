#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 			

typedef struct {
	unsigned char * h_in_out[2];
	unsigned char * data_CEDD, *out_CEDD, *theta_CEDD;
	
	int nRows;
	int nCols;
	int gridDimX;
	int gridDimY;
	int *zc_slc;
} t_CEDD_params;

//Gaussian
int launch_preemp_GCEDD(void *kstub);
int launch_orig_GCEDD(void *kstub);
int launch_slc_GCEDD(void *kstub);
int GCEDD_start_kernel(void *arg);
int GCEDD_end_kernel(void *arg);
int GCEDD_start_mallocs(void *arg);
int GCEDD_start_transfers(void *arg);

__global__ void
original_gaussianCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD, 
						int gridDimY, int iter_per_subtask);

//Sobel
int launch_preemp_SCEDD(void *kstub);
int launch_orig_SCEDD(void *kstub);
int launch_slc_SCEDD(void *kstub);
int SCEDD_start_kernel(void *arg);
int SCEDD_end_kernel(void *arg);
int SCEDD_start_mallocs(void *arg);
int SCEDD_start_transfers(void *arg);

__global__ void
original_sobelCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD, int iter_per_subtask);

//Non max supp
int launch_preemp_NCEDD(void *kstub);
int launch_orig_NCEDD(void *kstub);
int launch_slc_NCEDD(void *kstub);
int NCEDD_start_kernel(void *arg);
int NCEDD_end_kernel(void *arg);
int NCEDD_start_mallocs(void *arg);
int NCEDD_start_transfers(void *arg);

__global__ void
original_nonCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD, int iter_per_subtask);

//Hyst
int launch_preemp_HCEDD(void *kstub);
int launch_orig_HCEDD(void *kstub);
int launch_slc_HCEDD(void *kstub);
int HCEDD_start_kernel(void *arg);
int HCEDD_end_kernel(void *arg);
int HCEDD_start_mallocs(void *arg);
int HCEDD_start_transfers(void *arg);

__global__ void
original_hystCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD, int iter_per_subtask);

//CPU
void run_cpu_threads(unsigned char *buffer0, unsigned char *buffer1, unsigned char *theta, int rows, int cols, int num_threads, int t_index);