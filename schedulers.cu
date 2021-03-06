#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include <cuda_profiler_api.h>

typedef struct{
	double start_time;
	double end_time;
}t_ktime;


// Applications is composed of one or several kernels
typedef struct{
	int num_kernels;
	int index;
	t_Kernel kid[8]; // Max: 8 kernels per application
	t_kernel_stub* kstubs[8]; // One kernel stub per kernel
}t_application;


// Tables to store results for solo exectuions
t_smk_solo *smk_solo; // 
t_smt_solo *smt_solo;

// Tables to store coexecution results
t_smk_coBlocks **smk_conc;
t_smt_coBlocks **smt_conc; //tpms of each kernel in coexection

// Table to store better speedups in coexecution

t_co_speedup **smk_best_sp;
t_co_speedup **smt_best_sp;

int read_profling_tables()
{
	FILE *fp;
	
	if ((fp = fopen("profiling_table.bin", "r")) == NULL) {
		printf("Cannot read file\n");
		return -1;
	}
	
	// Number of kernels 
	int n = Number_of_Kernels-1;
	fread (&n, 1, sizeof(int), fp);
	
	// Create t_smk_solo smk_info_solo[]
	smk_solo = (t_smk_solo *)calloc(n, sizeof(t_smk_solo));
	
	// Load t_smk_solo smk_solo[]
	for (int i=0; i<n; i++){
		fread(&smk_solo[i].num_configs, 1, sizeof(int), fp);
		smk_solo[i].tpms = (double *)calloc(smk_solo[i].num_configs, sizeof(double));
		fread(smk_solo[i].tpms, smk_solo[i].num_configs, sizeof(double), fp);
	}
	
	// Create t_smt_solo smt_solo[]
	smt_solo = (t_smt_solo *)calloc(n, sizeof(t_smt_solo));
	
	// Load t_smt_solo smt_info_solo
	for (int i=0; i<n; i++){
		fread(&smt_solo[i].num_configs, 1, sizeof(int), fp);
		smt_solo[i].tpms = (double *)calloc(smt_solo[i].num_configs, sizeof(double));
		fread(smt_solo[i].tpms, smt_solo[i].num_configs, sizeof(double), fp);
	}
	
	// Create t_smk_coBlocks smk_conc

	smk_conc = (t_smk_coBlocks **)calloc(n, sizeof(t_smk_coBlocks *));
	for (int i=0; i<n; i++)
		smk_conc[i] = (t_smk_coBlocks *)calloc(n, sizeof(t_smk_coBlocks));
	
	//Load t_smk_coBlocks smk_conc
	
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fread(smk_conc[i][j].kid, 2, sizeof(t_Kernel), fp);
			fread(&smk_conc[i][j].num_configs, 1, sizeof(int), fp);
			
			smk_conc[i][j].pairs = (int **)calloc(smk_conc[i][j].num_configs, sizeof(int *));
			for (int k=0; k<smk_conc[i][j].num_configs; k++)
				smk_conc[i][j].pairs[k] = (int *)calloc(2, sizeof(int));
			for (int k=0; k<smk_conc[i][j].num_configs; k++)
				fread(smk_conc[i][j].pairs[k], 2, sizeof(int), fp);
			
			smk_conc[i][j].tpms = (double **)calloc(smk_conc[i][j].num_configs, sizeof(double *));
			for (int k=0; k<smk_conc[i][j].num_configs; k++)
				smk_conc[i][j].tpms[k] = (double *)calloc(2, sizeof(double));
			for (int k=0; k<smk_conc[i][j].num_configs; k++)
				fread(smk_conc[i][j].tpms[k], 2, sizeof(double), fp);
		}
		
	// Create t_smt_coBlocks smt_conc

	smt_conc = (t_smt_coBlocks **)calloc(n, sizeof(t_smt_coBlocks *));
	for (int i=0; i<n; i++)
		smt_conc[i] = (t_smt_coBlocks *)calloc(n, sizeof(t_smt_coBlocks));
	
	//Load t_smt_coBlocks smt_conc
	
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fread(smt_conc[i][j].kid, 2, sizeof(t_Kernel), fp);
			fread(&smt_conc[i][j].num_configs, 1, sizeof(int), fp);
			
			smt_conc[i][j].pairs = (int **)calloc(smt_conc[i][j].num_configs, sizeof(int *));
			for (int k=0; k<smt_conc[i][j].num_configs; k++)
				smt_conc[i][j].pairs[k] = (int *)calloc(2, sizeof(int));
			for (int k=0; k<smt_conc[i][j].num_configs; k++)
				fread(smt_conc[i][j].pairs[k], 2, sizeof(int), fp);
			
			smt_conc[i][j].tpms = (double **)calloc(smt_conc[i][j].num_configs, sizeof(double *));
			for (int k=0; k<smt_conc[i][j].num_configs; k++)
				smt_conc[i][j].tpms[k] = (double *)calloc(2, sizeof(double));
			for (int k=0; k<smt_conc[i][j].num_configs; k++)
				fread(smt_conc[i][j].tpms[k], 2, sizeof(double), fp);
		}
		
	// Create t_co_speedup smk_best_sp
	
	smk_best_sp = (t_co_speedup **)calloc(n, sizeof(t_co_speedup *));
	for (int i=0; i<n; i++)
		smk_best_sp[i] = (t_co_speedup *)calloc(n, sizeof(t_co_speedup));
	
	// Load t_co_speedup smk_best_sp

	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fread(smk_best_sp[i][j].pairs, 2, sizeof(int), fp);
			fread(&smk_best_sp[i][j].speedup, 1, sizeof(double), fp);
		}
		
	// Create t_co_speedup smt_best_sp
	
	smt_best_sp = (t_co_speedup **)calloc(n, sizeof(t_co_speedup *));
	for (int i=0; i<n; i++)
		smt_best_sp[i] = (t_co_speedup *)calloc(n, sizeof(t_co_speedup));
	
	// Load t_co_speedup smt_best_sp

	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fread(smt_best_sp[i][j].pairs, 2, sizeof(int), fp);
			fread(&smt_best_sp[i][j].speedup, 1, sizeof(double), fp);
		}
		
	fclose(fp);
	
	return 0;
}


int alloc_HtD_tranfers(t_application *applications, int num_applications)
{
	for (int i=0; i<num_applications; i++) {
		for (int j=0; j < applications[i].num_kernels; j++){
			(applications[i].kstubs[j]->startMallocs)((void *)(applications[i].kstubs[j]));
			(applications[i].kstubs[j]->startTransfers)((void *)(applications[i].kstubs[j]));
		}
	}
	return 0;
}

int nocke_all_applications(t_application *applications, int num_applications, t_ktime *ktime)
{	
	struct timespec now;
	
	cudaStream_t * exec_stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	
	//kstr->kstub->kconf.max_persistent_blocks = 1; // Only a block per SM will be launched by each stream
	int idSMs[2];
	idSMs[0]=0;
	// Launch streams
	for (int i=0; i<num_applications; i++) 
		for (int j=0; j < applications[i].num_kernels; j++) {
			idSMs[1] = applications[i].kstubs[j]->kconf.numSMs-1;
			applications[i].kstubs[j]->idSMs = idSMs;
			/*kstub->execution_s = exec_stream;

			
			int cont_task;
			t_kernel_stub *kstub = applications[i].kstubs[j];
			cudaMemcpy(&cont_task, kstub->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost);

			State state[MAX_STREAMS_PER_KERNEL];
			cudaMemcpy(state, kstub->gm_state,  sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaMemcpyDeviceToHost);
			printf("Kid= %d, cont=%d, Status=%d, numSMs=%d, Max_pers=%d, minSM=%d, maxSM=%d ttasks=%d coar=%d index=%d atream=%lld\n", kstub->id, cont_task, state[0], kstub->kconf.numSMs, 
			kstub->kconf.max_persistent_blocks,kstub->idSMs[0],kstub->idSMs[1],kstub->total_tasks,
						kstub->kconf.coarsening, kstub->stream_index, kstub->execution_s);
			*/
			(applications[i].kstubs[j]->launchCKEkernel)(applications[i].kstubs[j]);
			
			clock_gettime(CLOCK_REALTIME, &now);
			ktime[applications[i].kstubs[j]->id].start_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9; 
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_REALTIME, &now);
			ktime[applications[i].kstubs[j]->id].end_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9; 
		}
	
	// Reset task counter
	
	for (int i=0; i<num_applications; i++)
		for (int j=0; j < applications[i].num_kernels; j++) 
			cudaMemset(applications[i].kstubs[j]->d_executed_tasks, 0, sizeof(int));
	
	return 0;
}


// Assign a kstub to a kstreams already create;
int assing_kstreams(t_kernel_stub *kstub, t_kstreams *kstr)
{
	kstr->kstub = kstub;
	kstr->num_streams = 0;
	kstr->save_cont_tasks = 0;
	
	return 0;
}

// Given a kernel seach for the best partner (highest speeup in coexec) from the kernel ready list
int new_get_best_partner(t_Kernel curr_kid, t_Kernel *kid, State *k_done, float **bad_partner, int num_applications, t_Kernel *select_kid, int *select_index, int *b0, int *b1)
{

	double best_perf = -1.0;
	int best_index;
	t_Kernel best_kid;
	
	for (int i=0; i<num_applications; i++){ // For the remainning kernels 
		if (k_done[i] == READY){ // If kernel has not been executed 
			t_co_speedup *info = &smk_best_sp[curr_kid][kid[i]];
			if (info->speedup > best_perf) { // Search for best partnet (highest speedup in coexec) among ready kernels
				best_perf = info->speedup;
				best_index = i;
				best_kid = kid[i];
			}
		}
	}
			
	if (best_perf >=MIN_SPEEDUP) {
		*select_kid = best_kid;
		*select_index = best_index;
		*b0 = smk_best_sp[curr_kid][best_kid].pairs[0];
		*b1 = smk_best_sp[curr_kid][best_kid].pairs[1];
	}else{
		*select_kid = EMPTY; // Indicate no coexecution 
		*b0 = smk_solo[curr_kid].num_configs; // If performace is low the running kernel is executed with all the blocks
	}
	
	return 0;
}

int new_find_first_kernel( State *k_done, int *index, int num_kernels)
{
	int i;
	for (i=0;i<num_kernels; i++){
		if (k_done[i] == READY) {
			*index = i;
			return 0;
		}
	}
	
	*index = -1; // No available kernel found
	
	return 0;
}

// greedy_scheduler is oriented to reduce the makespan of a set of applications
// Assumning a list of ready kernels, this scheduler selects a pair of kernels to be coexecuted
// First pair selection is based on the kernels that achived the highest speedup. W
// Then when one of the two kernels finishe, the next ready one having tje kighest speepup when coexecutued with the alreadu running is selected.
// Partial evition and adding of streams is performed to establshed the adequated resource assignement to each running kernel 


int greedy_coexecution(int deviceId)
{	
	cudaError_t err;
	struct timespec now;

	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);

	// Create sched structure
	
	t_sched sched;
	create_sched(&sched);
	
	// Load profilinf tables
	read_profling_tables();
	
	// Important for reducing the number od streams launched by RCONV and CCONV
	// Profiling tables are modified for RCONV to reduce the number of concurrent streams
	int n = Number_of_Kernels-1;
	for (int i=0; i<n; i++) {
		smk_best_sp[RCONV][i].pairs[0] = smk_best_sp[RCONV][i].pairs[0]/2;
		smk_best_sp[i][RCONV].pairs[1] = smk_best_sp[i][RCONV].pairs[1]/2;
	}
	smk_solo[RCONV].num_configs = smk_solo[RCONV].num_configs /2;
	
	// Aplications
	int num_applications=13;
	t_application *applications = (t_application *)calloc(num_applications, sizeof(t_application));
	
	applications[0].num_kernels = 1; applications[0].kid[0] = VA;
	applications[1].num_kernels = 1; applications[1].kid[0] = MM;
	applications[2].num_kernels = 1; applications[2].kid[0] = RCONV;
	applications[3].num_kernels = 1; applications[3].kid[0] = CCONV;
	applications[4].num_kernels = 1; applications[4].kid[0] = HST256;
	applications[5].num_kernels = 1; applications[5].kid[0] = Reduction;
	applications[6].num_kernels = 1; applications[6].kid[0] = PF;
	applications[7].num_kernels = 1; applications[7].kid[0] = BS;
	applications[8].num_kernels = 1; applications[8].kid[0] = SPMV_CSRscalar;
	applications[9].num_kernels = 1; applications[9].kid[0] = GCEDD; 
	applications[10].num_kernels = 1; applications[10].kid[0] = SCEDD; 
	applications[11].num_kernels = 1; applications[11].kid[0] = NCEDD; 
	applications[12].num_kernels = 1; applications[12].kid[0] = HCEDD;
	
									 
	
	/*
	applications[0].num_kernels = 4; applications[0].kid[0] = GCEDD;
									 applications[0].kid[1] = SCEDD; 
									 applications[0].kid[2] = NCEDD; 
									 applications[0].kid[3] = HCEDD;
	applications[1].num_kernels = 2; applications[1].kid[0] = RCONV;
									 applications[1].kid[1] = CCONV;		
	applications[2].num_kernels = 1; applications[2].kid[0] = HST256;
	applications[3].num_kernels = 1; applications[3].kid[0] = Reduction;
	applications[4].num_kernels = 1; applications[4].kid[0] = PF;
	applications[5].num_kernels = 1; applications[5].kid[0] = VA;
	applications[6].num_kernels = 1; applications[6].kid[0] = BS;
	applications[7].num_kernels = 1; applications[7].kid[0] = SPMV_CSRscalar;
	applications[8].num_kernels = 1; applications[8].kid[0] = MM; 
							 
*/
	
	// First kernel of each application in sent to ready
	
	t_Kernel *kid = (t_Kernel *) calloc(num_applications, sizeof(t_Kernel)); // List of ready kernels
	for (int i=0; i<num_applications; i++) 
		kid[i] = applications[i].kid[0];
	
	// k_done annotates kernel state
	State *k_done = (State *)calloc(num_applications, sizeof(int));

	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	/** Create stubs ***/
	for (int i=0; i<num_applications; i++) 
		for (int j=0; j < applications[i].num_kernels; j++)
			if (j == 0) // If first applicacion kernel
				create_stubinfo(&applications[i].kstubs[j], deviceId, applications[i].kid[j], transfers_s, &preemp_s);
			else
				create_stubinfo_with_params(&applications[i].kstubs[j], deviceId, applications[i].kid[j], transfers_s, &preemp_s, applications[i].kstubs[0]->params);
	
	// Make allocation and HtD transfer for applications
	alloc_HtD_tranfers(applications, num_applications);
	cudaDeviceSynchronize();
	
	// Bad patners: each kenel annotates if of partner with bad speedpup in coexecution
	float **bad_partner = (float **)calloc(Number_of_Kernels, sizeof(float *));
	for (int i=0;i<Number_of_Kernels; i++)
		bad_partner[i] = (float *)calloc(Number_of_Kernels, sizeof(float));
	
	// Annotate start and end kernel execution time
	t_ktime *ktime_conc = (t_ktime *)calloc(Number_of_Kernels-1, sizeof(t_ktime));
	// Annotate start and end kernel execution time
	t_ktime *ktime2_seq = (t_ktime *)calloc(Number_of_Kernels-1, sizeof(t_ktime));
	
	// Save application to change the order in following iterations
	
	t_application *save_applications = (t_application *)calloc(num_applications, sizeof(t_application));
	for (int i=0; i<num_applications; i++){
		save_applications[i].num_kernels = applications[i].num_kernels;
		for (int j=0; j<applications[i].num_kernels; j++) {
			save_applications[i].kid[j] = applications[i].kid[j]; 
			save_applications[i].kstubs[j] = applications[i].kstubs[j]; 
		}
	}
	
	// Sequential execution
	t_ktime *ktime_seq = (t_ktime *)calloc(Number_of_Kernels-1, sizeof(t_ktime));
	clock_gettime(CLOCK_REALTIME, &now);
 	double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double start_seq_time = time1;
	nocke_all_applications(applications, num_applications, ktime_seq);
	clock_gettime(CLOCK_REALTIME, &now);
 	double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double seq_exec_time = time2-time1;
	
	for (int perm=0; perm<13; perm++) {
		
		// Application order permutation
		for (int i=0; i<num_applications; i++){
			applications[(i+perm) % num_applications].num_kernels = save_applications[i].num_kernels;
			applications[(i+perm) % num_applications].index=0;
			for (int j=0; j<save_applications[i].num_kernels; j++) {
				applications[(i+perm) % num_applications].kid[j] = save_applications[i].kid[j]; 
				applications[(i+perm) % num_applications].kstubs[j] = save_applications[i].kstubs[j]; 
			}
		}
		
		for (int i=0; i<num_applications; i++) 
			kid[i] = applications[i].kid[0];

		
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(num_applications, sizeof(t_kstreams));
	for (int i=0; i<num_applications; i++)
		create_kstreams(applications[i].kstubs[0], &kstr[i]);
	
	//  Initiallu all kernels are ready
	for (int i=0; i< num_applications; i++)
		k_done[i] = READY;
	
	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);

	// Launch proxy
	launch_generic_proxy((void *)&sched);	// Launch proxy
	
	// Select initial kernel
	int task_index = 0; // Index of the kernel in the array with ready kernels;
	//k_done[task_index] = 1; // Kernel removed from pending kernels*/
	
	// Reset timers
	memset(ktime_conc, 0, sizeof(t_ktime)*(Number_of_Kernels-1));
	 
	int kernel_idx; // Position of kernel in coexec struc
	double speedup; 
	
	clock_gettime(CLOCK_REALTIME, &now);
 	time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double time_sample;
	do {
		
		if (coexec.num_kernels == 0) {
			if (perm==1) 
				printf("Aqui\n");
			new_find_first_kernel(k_done, &task_index, num_applications); // Index in k_done: Arbitrarily we choode the first ready one from the array head
			if (task_index == -1)
				break; // Exit: no remaining kernels*/
			k_done[task_index] = RUNNING;
		}
			
		// Given kid[task_index] kernel, choose the partner with highest performance 
		int task_index2; // index in kernel ready list of the slected kernel
		int b0, b1;
		t_Kernel select_kid;
	
		new_get_best_partner(kid[task_index], kid, k_done, bad_partner, num_applications, &select_kid, &task_index2, &b0, &b1);
		
		char k0_name[30]; char k1_name[30];
		kid_from_index(kid[task_index], k0_name);
		kid_from_index(select_kid, k1_name);		
		//printf("---> Selecting %s(%d) %s(%d)\n",k0_name, b0, k1_name, b1 );   
		
		// Ckeck if kernel is already in coexec (because it is executing)
		int pos, dif;
		kernel_in_coexec(&coexec, &kstr[task_index], &pos);
		
		// kernel position in coexec struct (0 or 1)
		if (pos == -1) // kernel is not in coexec
			add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index], b0, task_index); // Add b0 streams
		else {
			if ((dif = ( b0 - kstr[task_index].num_streams)) > 0) // New streams must be added
				add_streams_to_kernel(&coexec, &sched, coexec.kstr[pos], dif);
			else {
				evict_streams(coexec.kstr[pos], -dif); // Some running streams must be evicted
				coexec.num_streams[pos] +=dif;
			}
		}
		
		if (select_kid != EMPTY){ // if coexecution is theorically benefical
			k_done[task_index2] = RUNNING; // Romove kernel from ready list
			add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index2], b1, task_index2); // Add b0 streams
		}
		
		// Execute kernels (launching streams) in coexec structure
		launch_coexec(&coexec);	
		
		// Annotate kernel start time
		clock_gettime(CLOCK_REALTIME, &now);
		time_sample = (double)now.tv_sec+(double)now.tv_nsec*1e-9;	
		if (coexec.kstr[0] != NULL)
			if (ktime_conc[coexec.kstr[0]->kstub->id].start_time == 0)
				ktime_conc[coexec.kstr[0]->kstub->id].start_time = time_sample;
		if (coexec.kstr[1] != NULL)
			if (ktime_conc[coexec.kstr[1]->kstub->id].start_time == 0)
				ktime_conc[coexec.kstr[1]->kstub->id].start_time = time_sample;
		
		// Wait for termination condition
	
		wait_for_kernel_termination_with_proxy(&sched, &coexec, &kernel_idx, &speedup);
		
		//if (coexec.num_kernels == 1) 
		//	break; // The last kernels has finished. Exit
	
		if (speedup < MIN_SPEEDUP && coexec.num_kernels == 2){	// If speedup is not good, stop second kernel
			
			evict_streams(coexec.kstr[1], coexec.kstr[1]->num_streams); // Stop all the streams of the second kernel (why not the first one?-> criterion based on remaining execution time?)
			k_done[coexec.queue_index[1]]= READY;  //Put the second kernel as ready again
			
			//bad_partner[coexec.kstr[0]->kstub->id][coexec.kstr[1]->kstub->id] = -1;//speedup; // Annotate bad partner
			//bad_partner[coexec.kstr[1]->kstub->id][coexec.kstr[0]->kstub->id] = -1; //;
			
			clock_gettime(CLOCK_REALTIME, &now);
			time_sample = (double)now.tv_sec+(double)now.tv_nsec*1e-9;	
			ktime_conc[coexec.kstr[1]->kstub->id].end_time = time_sample;
			rem_kernel_from_coexecution(&coexec, &sched, coexec.kstr[1]); //Remove second kernel for coexec struct
			
			
			// Add new exectuing streams to first kernel (in coexec struct) so that it will run the maximum number of streams
			add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], smk_solo[coexec.kstr[0]->kstub->id].num_configs - coexec.kstr[0]->num_streams);
			
			launch_coexec(&coexec); // Launch new streams of first kernel
			
			wait_for_kernel_termination_with_proxy(&sched, &coexec, &kernel_idx, &speedup); // Wait first kernel to finish, kernel_idx.->index in coexec
			
			int kind = coexec.queue_index[0]; // Save index in ready list of the finished kernel
			
			// Update coexec: remove first kernel
			clock_gettime(CLOCK_REALTIME, &now);
			time_sample = (double)now.tv_sec+(double)now.tv_nsec*1e-9;		
			ktime_conc[coexec.kstr[0]->kstub->id].end_time = time_sample;
			rem_kernel_from_coexecution(&coexec, &sched, coexec.kstr[0]);
			
			// If application has more kernels activate the next one
			 // Kernel index in ready list
	
			if (applications[kind].index + 1 < applications[kind].num_kernels) {
				applications[kind].index++;
				kid[kind] = applications[kind].kid[applications[kind].index]; // get ID of new kernel
				k_done[kind] = READY; // Set ready
				assing_kstreams(applications[kind].kstubs[applications[kind].index], &kstr[kind]);
			}
			else
				k_done[kind] = DONE; // Otherwise, application has finished
		
		}
		else
		{
			
			int kind = coexec.queue_index[kernel_idx]; // Save index in ready list of the finished kernel

			// Remove finished kernel
			clock_gettime(CLOCK_REALTIME, &now);
			time_sample = (double)now.tv_sec+(double)now.tv_nsec*1e-9;		
			ktime_conc[coexec.kstr[kernel_idx]->kstub->id].end_time = time_sample;

			rem_kernel_from_coexecution(&coexec, &sched, coexec.kstr[kernel_idx]);
			
			// If application has more kernels activate the next one
			if (applications[kind].index + 1 < applications[kind].num_kernels) {
				applications[kind].index++;
				kid[kind] = applications[kind].kid[applications[kind].index]; // get ID of new kernel and write in in kid list
				k_done[kind] = READY; // Set the new kernel ready
				assing_kstreams(applications[kind].kstubs[applications[kind].index], &kstr[kind]); // Assing new kernel to application kstreams
			}
			else
				k_done[kind] = DONE; // Otherwise, application has finished
		}
		
		if (coexec.num_kernels != 0){
			// find task index of the running kernel
			int i;
			for (i=0;i<MAX_NUM_COEXEC_KERNELS; i++)
				if (coexec.kstr[i] != NULL)
					task_index = coexec.queue_index[i];
		}
	
	} while (1);
	

	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_REALTIME, &now);
 	time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double conc_exec_time = time2-time1;
	//printf("Concurrent excution time=%f sec.\n", time2-time1);
	
	// Free
	remove_coexec(&coexec);
	for (int i=0; i<num_applications; i++)
		remove_kstreams(&kstr[i]);
	free(kstr);
	
	// Reset task counter
	
	for (int i=0; i<num_applications; i++)
		for (int j=0; j < applications[i].num_kernels; j++) 
			cudaMemset(applications[i].kstubs[j]->d_executed_tasks, 0, sizeof(int));
		
	// Set streams status to PREP
	for (int i=0; i<num_applications; i++)
		for (int j=0; j < applications[i].num_kernels; j++){
			for (int k=0; k<MAX_STREAMS_PER_KERNEL; k++)
				applications[i].kstubs[j]->h_state[k] = PREP;
			cudaMemcpy(applications[i].kstubs[j]->gm_state,  applications[i].kstubs[j]->h_state, sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaMemcpyHostToDevice);
		}
			
	// Reset index 
	for (int i=0; i<num_applications; i++)
		for (int j=0; j < applications[i].num_kernels; j++)
			applications[i].kstubs[j]->stream_index = 0;
	
		
	// Calculate stats
	
	double acc_time=start_seq_time;
	for (int i=0;i<num_applications;i++) {
		for (int j=0; j < applications[i].num_kernels; j++){
			int id = applications[i].kstubs[j]->id;
			ktime2_seq[id].start_time =acc_time;
			acc_time += (ktime_seq[id].end_time - ktime_seq[id].start_time);
			ktime2_seq[id].end_time =acc_time;
		}
	}
		
	
	double antt_seq=0, antt_conc=0;
	int cont =0;
	printf("Kid \t endtime_seq \t endtime_conc \t NTT_Seq \t NTT_conc\n");
	for (int i=0; i<Number_of_Kernels-1; i++)
		if (ktime_seq[i].start_time != 0){
			printf("%d \t %f \t %f \t %f \t %f\n", i, (ktime2_seq[i].end_time - start_seq_time), (ktime_conc[i].end_time - time1), (ktime2_seq[i].end_time - start_seq_time) /(ktime2_seq[i].end_time - ktime2_seq[i].start_time),  (ktime_conc[i].end_time - time1)/(ktime2_seq[i].end_time - ktime2_seq[i].start_time) );
			antt_seq += (ktime2_seq[i].end_time - start_seq_time) /(ktime_seq[i].end_time - ktime_seq[i].start_time);
			antt_conc += (ktime_conc[i].end_time - time1)/(ktime_seq[i].end_time - ktime_seq[i].start_time);
			cont++;
		}
		
	printf("ANTT_seq \t ANTT_conc\n");
	printf("%f \t %f\n", antt_seq/(double)cont, antt_conc/(double)cont);
	printf("Speepup = %f\n", seq_exec_time/conc_exec_time);
	
	}
	
	return 0;
}

int rt_scheduler(int deviceId, double max_slowdown, t_Kernel bg_kid)
{
	struct timespec now;
    cudaError_t err;
	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);

	// Load profilinf tables
	read_profling_tables();
	
	// Create sched structure
	t_sched sched;
	create_sched(&sched);
	
	// Aplications
	int num_applications=6;
	t_application *applications = (t_application *)calloc(num_applications, sizeof(t_application));
	
	applications[0].num_kernels = 4; applications[0].kid[0] = GCEDD;
									 applications[0].kid[1] = SCEDD; 
									 applications[0].kid[2] = NCEDD; 
									 applications[0].kid[3] = HCEDD;
	applications[1].num_kernels = 1; applications[1].kid[0] = bg_kid;	
	applications[2].num_kernels = 1; applications[2].kid[0] = bg_kid;	
	applications[3].num_kernels = 1; applications[3].kid[0] = bg_kid;	
	applications[4].num_kernels = 1; applications[4].kid[0] = bg_kid;
	applications[5].num_kernels = 1; applications[5].kid[0] = bg_kid;	
	
	// First kernel of each application in sent to ready
	
	t_Kernel *kid = (t_Kernel *) calloc(num_applications, sizeof(t_Kernel)); // List of ready kernels
	for (int i=0; i<num_applications; i++) 
		kid[i] = applications[i].kid[0];
	
	
	// k_done annotates kernel state
	State *k_done = (State *)calloc(num_applications, sizeof(int));
	for (int i=0; i< num_applications-1; i++) //App SCEDD is latency sensitive
		k_done[i] = READY;

	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	/** Create stubs ***/
	for (int i=0; i<num_applications; i++) 
		for (int j=0; j < applications[i].num_kernels; j++)
			if (j == 0) // If first applicacion kernel
				create_stubinfo(&applications[i].kstubs[j], deviceId, applications[i].kid[j], transfers_s, &preemp_s);
			else
				create_stubinfo_with_params(&applications[i].kstubs[j], deviceId, applications[i].kid[j], transfers_s, &preemp_s, applications[i].kstubs[0]->params);
	
	// Make allocation and HtD transfer for applications
	alloc_HtD_tranfers(applications, num_applications);
	cudaDeviceSynchronize();

	// Calculate sequential execution time (overlapping is still possible)
    clock_gettime(CLOCK_REALTIME, &now);
 	double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	t_ktime *ktime = (t_ktime *)calloc(Number_of_Kernels-1, sizeof(t_ktime));
	nocke_all_applications(applications, num_applications, ktime);
	clock_gettime(CLOCK_REALTIME, &now);
 	double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	printf("Sequential execution time =%f sec\n", time2-time1);

	double seq_time = ktime[GCEDD].end_time - ktime[GCEDD].start_time 
					+ ktime[SCEDD].end_time - ktime[SCEDD].start_time
					+ ktime[NCEDD].end_time - ktime[NCEDD].start_time
					+ ktime[HCEDD].end_time - ktime[HCEDD].start_time	;
					
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(num_applications, sizeof(t_kstreams));
	for (int i=0; i<num_applications; i++)
		create_kstreams(applications[i].kstubs[0], &kstr[i]);

	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);
	
	// Launch proxy
	launch_generic_proxy((void *)&sched);	// Launch proxy
	 
	int kernel_idx; // Position of kernel in coexec struc
	double speedup; 

	int task_index = 0; // Index of kernel
	int task_index2 = 1;
	double time1_rt, time2_rt;
	clock_gettime(CLOCK_REALTIME, &now);
 	time1_rt = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double acc_numstreams_time=0; // to calculate time · num_streams 
	
	do {	
	
		// Coexecution configuration with th maximum number of streams for kernel 0
		t_Kernel idk0 = kstr[task_index].kstub->id;
		t_Kernel idk2 = kstr[task_index2].kstub->id;
		int num_cf = smk_conc[idk0][idk2].num_configs;
		int b0, b1; // Max num of streams must be assigned to ls kernel
		if (smk_conc[idk0][idk2].pairs[0][0] > smk_conc[idk0][idk2].pairs[0][1]) {
			b0 = smk_conc[idk0][idk2].pairs[0][0];
			b1 = smk_conc[idk0][idk2].pairs[0][1];
		} 
		else {
			int num_cf = smk_conc[idk0][idk2].num_configs;
			b0 = smk_conc[idk0][idk2].pairs[num_cf-1][0];
			b1 = smk_conc[idk0][idk2].pairs[num_cf-1][1];
		} 
		
		printf("kids=(%d,%d) streams=(%d,%d)\n", idk0, idk2, b0, b1);
		
	/*	printf("Max tmps=%f\n", smk_conc[idk0][idk2].tpms[0][0]);
		for (int i=0;i<=7;i++)
			printf("Solo num_str=%d tpms=%f\n", i+1, smk_solo[idk0].tpms[i]);

		for (int i=0;i<7;i++)
			printf("Coexec num_str=%d tpms=%f\n", i+1, smk_conc[idk0][idk2].tpms[i][0]);		
		*/
		int pos0, pos1;
		kernel_in_coexec(&coexec, &kstr[task_index], &pos0);
		kernel_in_coexec(&coexec, &kstr[task_index2], &pos1);
		
		if (pos0 == -1 && pos1 == -1) { // No kernels are executiing
			add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index], b0, task_index); // Add b0 streams
			add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index2], b1, task_index2); // Add b1 streams
		}
		else {
			if (pos0 == -1){
				if (coexec.kstr[1]->num_streams > b1){ // If non-ls kernel is running and have too much streams
					int rem_streams = coexec.kstr[1]->num_streams - b1;
					evict_streams(coexec.kstr[1], rem_streams); // Now, non-ls kernels has b1 streams
					coexec.num_streams[1] -= rem_streams;
					printf("Borrando %d(%d)streams de kernel %d\n", rem_streams, coexec.num_streams[1], coexec.kstr[1]->kstub->id);
				}
				add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index], b0, task_index); // Add b0 streams
			}
				
			if (pos1 == -1){
				if (coexec.kstr[0]->num_streams < b0) // If LS kernel is running and has too few streams
					add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], b0 - coexec.kstr[0]->num_streams); // Add more streams to LS kernel
				add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index2], b1, task_index2); // Add b1 streams
			}	
		}
		
		// Execute kernels (launching streams) in coexec structure
		launch_coexec(&coexec);
		
		double start_time;
		if (pos0 == -1) {// When a new ls kernel is launched, get time
			// Get current time
			clock_gettime(CLOCK_REALTIME, &now);
			start_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		}
		
		// Calculate minimum tpms that ls kernel can obtain
		double max_tpms = smk_solo[idk0].tpms[smk_solo[idk0].num_configs-1]; // Faster tpms
		double min_tpms = max_slowdown * max_tpms;
		 
		// Wait for termination condition
		int rc;
		double sampling_interval = 0.0004;
		int flag_rc2 = 0, stop_pingpong=0;
		do {
			ls_coexec_wait_for_kernel_termination_with_proxy(&sched, &coexec, min_tpms, start_time, sampling_interval, &acc_numstreams_time, &kernel_idx, &rc);
			printf("rc=%d\n", rc);
			if (rc == 1 && coexec.kstr[1] != NULL) { // LS kernel performace is low: steal a stream of non LS kernel if this kernel is still running
				evict_streams(coexec.kstr[1], 1);
				coexec.num_streams[1]--;
				if (coexec.kstr[1]->num_streams == 0) {
					rem_kernel_from_coexecution(&coexec, &sched, coexec.kstr[1]);
					printf("Eliminando stream de no-ls\n");
				}
				add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], 1);
				launch_coexec(&coexec);
				if (flag_rc2 == 1) stop_pingpong =1;
			}
			if (rc == 2 && coexec.kstr[0]->num_streams >= 2 && stop_pingpong == 0) { // LS kernel performace is high: give a stream to non LS kerne LS Kernel has, at least 2, streams
				evict_streams(coexec.kstr[0], 1);
				coexec.num_streams[0]--;
				kernel_in_coexec(&coexec, &kstr[task_index2], &pos1);
				if (pos1 != -1) // If non ls is running
					add_streams_to_kernel(&coexec, &sched, coexec.kstr[1], 1); // add a new stream to non ls
				else
					add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index2], 1, task_index2); // else add nonls to coexec with one stream
				launch_coexec(&coexec);
				flag_rc2=1;
			}

			sampling_interval = 0.0002;
				
		}
		while (rc == 1 || rc==2); // If rc==0, a kernel has finished 
		
		int kind = coexec.queue_index[kernel_idx]; // Save index in ready list of the finished kernel

		// Remove finished kernel
		rem_kernel_from_coexecution(&coexec, &sched, coexec.kstr[kernel_idx]);
		printf("Eliminando kernel con idx = %d\n", kernel_idx);
		
		if (kid[kind] == HCEDD)
			break;
		
		// If application has more kernels activate the next one	
		if (applications[kind].index + 1 < applications[kind].num_kernels) {
			applications[kind].index++;
			kid[kind] = applications[kind].kid[applications[kind].index]; // get ID of new kernel
			k_done[kind] = READY; // Set ready
			assing_kstreams(applications[kind].kstubs[applications[kind].index], &kstr[kind]);
		}
		else { 
		
			k_done[kind] = DONE; // Set ready
			task_index2++;  // move to nex non-ls kernel
			
			if (task_index2 >= num_applications){
				printf("Error: no hay suficiente sapplicaciones de non ls kernel. Termino de forma anticipada\n");
				break;
			}
				
		}
				
	
	} while (1);
	
	clock_gettime(CLOCK_REALTIME, &now);
 	time2_rt = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	double rt_exec_time = time2_rt - time1_rt;
	printf("kid, Slowdownm Fullfil, weighted_streams_per_time, \n");
	printf("%d, %f, %f, %f\n", kid[1], max_slowdown, (max_slowdown*seq_time-rt_exec_time)/(max_slowdown*seq_time), acc_numstreams_time/rt_exec_time);
	//printf(",st=%f rt=%f Per=%f, num_streams=%f\n", seq_time, rt_exec_time, (max_slowdown*seq_time-rt_exec_time)/max_slowdown*seq_time, acc_numstreams_time/rt_exec_time);

	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	//printf("Concurrent excution time=%f sec.\n", time2_rt-time1_rt);
	
	
	return 0;
}

	

