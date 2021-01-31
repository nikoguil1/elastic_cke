#include <stdio.h>          /* printf()                 */
#include <stdlib.h>         /* exit(), malloc(), free() */
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>      /* key_t, sem_t, pid_t      */
#include <sys/shm.h>        /* shmat(), IPC_RMID        */
#include <sys/mman.h>		/* mmap						*/
#include <errno.h>          /* errno, ECHILD            */
#include <semaphore.h>      /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>          /* O_CREAT, O_EXEC          */
#include <pthread.h>
#include <time.h>

#include <cuda_profiler_api.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int change_thread_percentage(int percentage) {
	FILE *server_list = NULL;
	char server_string[256], command_string[256];
	int server_pid;
	server_list = popen("echo get_server_list | nvidia-cuda-mps-control", "r");
	if (!server_list)
	{
		perror("Error reading MPS server list");
		exit(-1);
	}
	fgets(server_string, 1000, server_list);
	while (!feof(server_list))
	{
		server_pid = atoi(server_string);
		fgets(server_string, 1000, server_list);
	}
	sprintf(command_string, "echo set_active_thread_percentage %d %d | nvidia-cuda-mps-control > /dev/null", server_pid, percentage);
	//printf("%s\n", command_string);
	int status = system(command_string);
	return(status);
}

int run_original(t_kernel_stub *kstub, double *exectime_s)
{
	//cudaEvent_t start, stop;
	//float elapsedTime;
	
	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);
	
	kstub->launchORIkernel(kstub);
	cudaDeviceSynchronize();
	
	//cudaEventCreate(&stop);
	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);

	//cudaEventElapsedTime(&elapsedTime, start, stop);
	
	//*exectime_s = (double)elapsedTime/1000;
	
	return 0;
}


typedef struct{
	t_kernel_stub **kstubs;
	int index; // Index in kstubs array
}t_args;

void *launch_app(void *arg)
{
	t_args *args;
	
	args = (t_args *)arg;
	int index = args->index;
	t_kernel_stub *kstub = args->kstubs[index];
	
	printf("Launching kid=%d\n", kstub->id);
	
	int deviceId = 2;
	cudaSetDevice(deviceId);
	
	double exec_time;
	run_original(kstub, &exec_time);
	
	 if (kstub->id == RCONV) {
		kstub = args->kstubs[index + 1];
		run_original(kstub, &exec_time);
	 }
	 
	 if (kstub->id == GCEDD) {
		kstub = args->kstubs[index + 1];
		run_original(kstub, &exec_time);
		
		kstub = args->kstubs[index + 2];
		run_original(kstub, &exec_time);
		
		kstub = args->kstubs[index + 3];
		run_original(kstub, &exec_time);
	 }
	 
	 pthread_exit(NULL);
}

int hyperQ_threads()
{
	
	t_Kernel kid[9];
	kid[0]=MM;
	kid[1]=VA;
	kid[2]=BS;
	kid[3]=Reduction;
	kid[4]=PF;
	kid[5]=GCEDD; // Ojo: en profiling.cu se procesan tambien los tres kernels restantes de la aplicacion
	kid[6]=SPMV_CSRscalar;
	kid[7]=RCONV; // Ojo: en profiling se procesa tambien CCONV
	kid[8]=HST256;
	
	int num_kernels = 2;
	
	// context and streams
	
	cudaError_t err;

	// Select device
	int deviceId = 2;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	}
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
		
	// Create kstbus
	int cont = 0;			
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(13, sizeof(t_kernel_stub*)); // 13 is the max number of kernels for all app
	
	int index[9];
	for (int i=0; i< num_kernels; i++) {
		
		index[i] = cont;
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		cont++;
		
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}
		
	// make HtD transfers of all kernels
	make_transfers(kstubs, cont);
	
	// Create threads to lauch app
	
	t_args args[9];
	for (int i=0; i<9; i++)
		args[i].kstubs = kstubs;
	pthread_t *thid = (pthread_t *) calloc(num_kernels, sizeof(pthread_t));
	for (int i=0; i<num_kernels; i++) {
		args[i].index = index[i];	
		pthread_create(&thid[i], NULL, launch_app, &args[i]);
	}
	
	for (int i=0; i<num_kernels; i++)
		pthread_join(thid[i], NULL);
	
	cudaDeviceSynchronize();
	
	return 0;
}
		

int main (int argc, char **argv)
{
	
	int it;                        /*      loop variables          */
    key_t shmkey;                 /*      shared memory key       */
    int shmid;                    /*      shared memory id        */
    sem_t *sem;                   /*      synch semaphore         *//*shared */
    pid_t pid;                    /*      fork pid                */
	pid_t childs_pid[2];
    int *p;                       /*      shared variable         *//*shared */
    unsigned int value;           /*      semaphore value         */
	
	double ProfilingTimeThreshold = 10.0; // Kernels are launched many times during this interval
//	hyperQ_threads();
//	return 0;


    /* initialize a shared variable in shared memory */
    shmkey = ftok ("/dev/null", 5);       /* valid directory name and a number */
    //printf ("shmkey for p = %d\n", shmkey);
    shmid = shmget (shmkey, sizeof (int), 0644 | IPC_CREAT);
    if (shmid < 0){                           /* shared memory error check */
        perror ("shmget\n");
        exit (1);
    }

    p = (int *) shmat (shmid, NULL, 0);   /* attach p to shared memory */
    *p = 0;
    //printf ("p=%d is allocated in shared memory.\n\n", *p);

    /********************************************************/

    /* initialize semaphores for shared processes */
    sem = sem_open ("pSem", O_CREAT | O_EXCL, 0644, 1); // Binary semaphore 
    /* name of semaphore is "pSem", semaphore is reached using this name */

    //printf ("semaphores initialized.\n\n");
	
	/*// kstubs
	
	cudaError_t err;

	// Select device
	int deviceId = 2;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	// Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); */

	t_Kernel kid[9];
	int index[9];
	kid[0]=MM;
	kid[1]=VA;
	kid[2]=BS;
	kid[3]=Reduction;
	kid[4]=PF;
	kid[5]=GCEDD; // Ojo: en profiling.cu se procesan tambien los tres kernels restantes de la aplicacion
	kid[6]=SPMV_CSRscalar;
	kid[7]=RCONV; // Ojo: en profiling se procesa tambien CCONV
	kid[8]=HST256;
	
	if ( argc > 2 ) kid[0] = kid_from_name(argv[2]);
	if ( argc > 3 )	kid[1] = kid_from_name(argv[3]);

	int num_kernels = 2;

	int percentage = 50;
	if ( argc > 4 ) percentage = atoi(argv[4]);
	change_thread_percentage(percentage);

	/*for (int i=0; i<num_kernels; i++){
		total_num_kernels++;
		if (kid[i] == RCONV) total_num_kernels++;
		if (kid[i] == GCEDD) total_num_kernels += 3;
	}*/
	
	/** Create stubs ***/
	// Ojo la lista de kernels sólo debe ponerse el primero de una aplicacion. Los demás
	// son creados por el siguiente código
	/*t_kernel_stub **kstubs = (t_kernel_stub **)calloc(total_num_kernels, sizeof(t_kernel_stub*));
	for (int i=0, cont=0; i<num_kernels; i++) {	
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		index[i] = cont;
		cont++;
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}

	// make HtD transfers of all kernels
	make_transfers(kstubs, total_num_kernels);*/
   
	void *message1 = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	void *message2 = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    /* fork child processes */
    for (it = 0; it < num_kernels; it++){
        pid = fork ();
        if (pid < 0) {
        /* check for error      */
            sem_unlink ("pSem");   
            sem_close(sem);  
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            printf ("Fork error.\n");
        }
        else if (pid == 0)
            break;                  /* child processes */
		childs_pid[it] = pid;
		char kname[100];
		kid_from_index(kid[it], kname);
		//printf("Child %d with PID %d will launch %s\n", it, childs_pid[it], kname);
    }

    /******************************************************/
    /******************   PARENT PROCESS   ****************/
    /******************************************************/
    if (pid != 0){
		unsigned long int numLaunchs[2], tmp0, tmp1, tmp2;
		float mps_time[2];
		float tmp3;
/*
		close( descr[1] ); // Close output descriptor
		read( descr[0], message1, 1000);
		sscanf(message1, "%d %d %lu %f", &tmp0, &tmp1, &tmp2, &tmp3);
		printf("Message1: %s <-> %d %d %lu %f\n", message1, tmp0, tmp1, tmp2, tmp3);
		if ( tmp0 == childs_pid[0] ) {		
			numLaunchs[0] = tmp2;
			mps_time[0] = tmp3;
		}
		else if ( tmp0 == childs_pid[1] ) {
			numLaunchs[1] = tmp2;
			mps_time[1] = tmp3;
		}
		read( descr[0], message2, 1000);
		sscanf(message2, "%d %d %lu %f", &tmp0, &tmp1, &tmp2, &tmp3);//		printf("Message2: %s <-> %d %d %f\n", message2, tmp1, tmp2, tmp3);
		printf("Message2: %s <-> %d %d %lu %f\n", message2, tmp0, tmp1, tmp2, tmp3);
		if ( tmp0 == childs_pid[0] ) {		
			numLaunchs[0] = tmp2;
			mps_time[0] = tmp3;
		}
		else if ( tmp0 == childs_pid[1] ) {
			numLaunchs[1] = tmp2;
			mps_time[1] = tmp3;
		}
		close( descr[0] );
*/
        /* wait for all children to exit */
		int retval;
        while (pid = waitpid (-1, &retval, 0)){
//			printf ("\n%d returns %d\n", pid, WEXITSTATUS(retval));
/*			if ( childs_pid[0] == pid )
				numLaunchs[0] = WEXITSTATUS(retval);
			else if ( childs_pid[1] == pid )
				numLaunchs[1] = WEXITSTATUS(retval);
*/
            if (errno == ECHILD)
                break;
        }

        //printf ("\nParent: All children have exited\n");
		sscanf((char *)message1, "%lu %lu %lu %f", &tmp0, &tmp1, &tmp2, &tmp3);
//		printf("Message1: %s <-> %lu %lu %lu %f\n", (char *)message1, tmp0, tmp1, tmp2, tmp3);
		if ( tmp0 == childs_pid[0] ) {		
			numLaunchs[0] = tmp2;
			mps_time[0] = tmp3;
		}
		else if ( tmp0 == childs_pid[1] ) {
			numLaunchs[1] = tmp2;
			mps_time[1] = tmp3;
		}

		sscanf((char *)message2, "%lu %lu %lu %f", &tmp0, &tmp1, &tmp2, &tmp3);//		printf("Message2: %s <-> %d %d %f\n", message2, tmp1, tmp2, tmp3);
//		printf("Message2: %s <-> %lu %lu %lu %f\n", (char *)message2, tmp0, tmp1, tmp2, tmp3);
		if ( tmp0 == childs_pid[0] ) {		
			numLaunchs[0] = tmp2;
			mps_time[0] = tmp3;
		}
		else if ( tmp0 == childs_pid[1] ) {
			numLaunchs[1] = tmp2;
			mps_time[1] = tmp3;
		}

        /* shared memory detach */
        shmdt (p);
        shmctl (shmid, IPC_RMID, 0);

        /* cleanup semaphores */
        sem_unlink ("pSem");   
        sem_close(sem);  
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */

		change_thread_percentage(100);

		// Select device
		cudaError_t err;
		int deviceId = 0;
		if ( argc > 1 ) deviceId = atoi(argv[1]);
		cudaSetDevice(deviceId);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceId);	
//		printf("Parent working on Device=%s\n", deviceProp.name);
	
		/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
		cudaStream_t *transfers_s;
		transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
		for (int i=0;i<2;i++){
			err = cudaStreamCreate(&transfers_s[i]);
			checkCudaErrors(err);
		}
	
		cudaStream_t preemp_s;
		checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
		
		double max_mps_time = 0, seq_time = 0;
		for ( it = 0; it < num_kernels; it++) {
			char kname[100];
			kid_from_index(kid[it], kname);	
			//printf("Parent: creating kstubs for kernel %s\n", kname);
	
			// Create kstbus
			int cont = 0;
			t_kernel_stub **kstubs = (t_kernel_stub **)calloc(4, sizeof(t_kernel_stub*)); // Four is the man number of kernels of a app
			int status = create_stubinfo(&kstubs[cont], deviceId, kid[it], transfers_s, &preemp_s);
			if ( status < 0 ) {
				printf("Exiting, no stubs created");
				exit(-1);
			}
			cont++;
		
			if (kid[it] == RCONV) { // RCONV params struct must be passed to CCONV 
				create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
				cont++;
			}		
			if (kid[it] == GCEDD) {
				create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
				cont++;
				create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
				cont++;			
				create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
				cont++;
			}
			
			// make HtD transfers of all kernels
			//printf("Parent: transfering data\n");
			make_transfers(kstubs, cont);
		
			// Solo original profiling
			//printf("Parent: Launching %lu times %s\n", numLaunchs[it], kname);
			double exectime_s[4];
			struct timespec now;
			double exec_time = 0.0;
			for (int n = 0; n < numLaunchs[it]; n++) {
				clock_gettime(CLOCK_MONOTONIC, &now);
				double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
				for (int i=0; i < cont; i++) {	
					run_original(kstubs[i], &exectime_s[i]);
				}
				clock_gettime(CLOCK_MONOTONIC, &now);
				double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;	
				exec_time += time2 - time1;
			}
			
			printf("%d\t%s\t%lu\t%f\t%f", percentage, kname, numLaunchs[it], mps_time[it], exec_time);
			if ( mps_time[it] > max_mps_time )
				max_mps_time = mps_time[it];
			seq_time += exec_time;
			if ( it == 1 )
				printf("\t%f\n", seq_time/max_mps_time );
			else
				printf("\n");
		}


        exit (0);
    }

    /******************************************************/
    /******************   CHILD PROCESS   *****************/
    /******************************************************/
    else{
		
		// context and streams
	
		cudaError_t err;

		// Select device
		int deviceId = 0;
		if ( argc > 1 ) deviceId = atoi(argv[1]);
		cudaSetDevice(deviceId);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceId);	
//		printf("Chidl %d working on Device=%s\n", it, deviceProp.name);
	
		/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
		cudaStream_t *transfers_s;
		transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
		for (int i=0;i<2;i++){
			err = cudaStreamCreate(&transfers_s[i]);
			checkCudaErrors(err);
		}
	
		cudaStream_t preemp_s;
		checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
		
//		printf("Child %d creating kstubs for kernel %d\n", it, kid[it]);
	
		// Create kstbus
		int cont = 0;
		t_kernel_stub **kstubs = (t_kernel_stub **)calloc(4, sizeof(t_kernel_stub*)); // Four is the man number of kernels of a app
		int status = create_stubinfo(&kstubs[cont], deviceId, kid[it], transfers_s, &preemp_s);
		if ( status < 0 ) {
			printf("Exiting, no stubs created");
			exit(-1);
		}
		cont++;
		
		if (kid[it] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[it] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
		
		// make HtD transfers of all kernels
		make_transfers(kstubs, cont);
		
//		printf("Child=%d Transferencia terminada\n", it);

		
		// Barrier
   /*     sem_wait (sem);           // P operation 
        printf ("  Child(%d) is in critical section.\n", it);
        //sleep (1);
        *p += 1 ;              //increment *p by 0, 1 or 2 based on i 
        printf ("  Child(%d) new value of *p=%d.\n", it, *p);
        sem_post (sem);           /// V operation 
     */   
		*p += 1;
		while (*p < num_kernels); // Spin lock

		
		// Solo original profiling
	
		char kname[100];
		kid_from_index(kstubs[0]->id, kname);
//		printf("Child %d launches %s (kid %d) \n", it, kname, kstubs[0]->id);
		double exectime_s[4];
		struct timespec now;
		clock_gettime(CLOCK_REALTIME, &now);
		double time0 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		double elapsed_time = 0.0, exec_time = 0.0;
		unsigned long int numLaunchs = 0;
		while ( elapsed_time < ProfilingTimeThreshold ) {
			for (int i=0; i < cont; i++) {	
				clock_gettime(CLOCK_REALTIME, &now);
				double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
				run_original(kstubs[i], &exectime_s[i]);
				clock_gettime(CLOCK_REALTIME, &now);
				double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;	
				exec_time += time2 - time1;
				elapsed_time = time2 - time0;
			}
			numLaunchs++;
		}


//		printf("Child %d ha lanzado %d veces el kernel %s : exectime=%f\n", it, numLaunchs, kname, exec_time);
		//printf("\t\t\tChild: %s launched %d times, exectime=%f\n", kname, numLaunchs, exec_time);
		char *child_message;
		if ( it == 0 ) child_message = (char *) message1;
		else  child_message = (char *) message2;
		sprintf(child_message, "%lu %lu %lu %f", (ulong) getpid(), (ulong) kid[it], (ulong) numLaunchs, exec_time);
//		sprintf((char *) message1, "%d %d %lu %f\n", getpid(), kid[it], numLaunchs, exec_time);
//		printf("%s <<>> %d %d %lu %f\n", child_message, getpid(), kid[it], numLaunchs, exec_time);
		/*
		if (kid[it] == GCEDD) {
			double exectime_s[4];
			for (int i=0; i < 4; i++) 
				run_original(kstubs[i], &exectime_s[i]);
		}
		else if (kid[index[it]] == RCONV) {
			double exectime_s[2];
			for (int i=0; i < 2; i++) 
				run_original(kstubs[i], &exectime_s[i]);
		}
		else {
			double exectime_s;
			run_original(kstubs[0], &exectime_s);
			printf("Child %d lanzando kernel %d. Tiempo=%f\n", it,  kid[it], exectime_s);
		}
		*/
		cudaDeviceSynchronize();
/*
		close( descr[0] ); // Close output descriptor
		write( descr[1], child_message, strlen(child_message));
		close( descr[1] );
*/
		exit(numLaunchs);
    }
}