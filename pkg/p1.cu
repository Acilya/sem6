
#include <stdio.h>
#include <time.h>

#define  N   		1000000
#define  BLOCK_SIZE	16    //threads per blocks

float 	   hArray[N];
float     *dArray;
int 	   blocks;
clock_t    cpu_startTime, cpu_endTime;
double     cpu_elapseTime = 0;
cudaEvent_t start, stop;
float      gpu_elapseTime = 0;


void prologue(void) {
   	cudaMalloc((void**)&dArray, sizeof(hArray));
   	cudaMemcpy(dArray, hArray, sizeof(hArray), cudaMemcpyHostToDevice);
}

void epilogue(void) {
	cudaMemcpy(hArray, dArray, sizeof(hArray), cudaMemcpyDeviceToHost);
	cudaFree(dArray);
	/*for(int i = 0; i < sizeof(hArray)/sizeof(float); i++) 
	{
		printf("%.1f\n", hArray[i]); 
	}*/
}

void cpu(float *A)
{
	float b;
	for (int x=0; x < N; x++)
	{
		b = A[x] * A[x] * A[x] + A[x] * A[x] + A[x];
		//printf("%.1f\n", b); 
	}
}


// Kernel
__global__ void pow3(float *A) {
	int x = blockDim.x * blockIdx.x + threadIdx.x; 

    if(x < N)
	    A[x] = A[x] * A[x] * A[x] + A[x] * A[x] + A[x]; 
}

int main(int argc, char** argv)
{
    int	 devCnt;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

    cudaGetDeviceCount(&devCnt);
    if(devCnt == 0) {
		perror("No CUDA devices available -- exiting.");
		return 1;
    }
	
	memset(hArray, 0, sizeof(hArray));
	for(int i = 0; i < N; i++) {
		hArray[i] =  i + 1;
	}	
	
	//CPU:
	cpu_startTime = clock();
	cpu(hArray);
	cpu_endTime = clock();
	cpu_elapseTime = ((cpu_endTime - cpu_startTime)/(double)CLOCKS_PER_SEC);
	printf("CPU time: %f ms\n", cpu_elapseTime);
	
	//GPU:
	prologue();
	cudaEventRecord(start, 0);
	//prologue();
    blocks = N / BLOCK_SIZE;   // amount of threads' blocks
    if(N % BLOCK_SIZE)
		blocks++;
    pow3<<<blocks, BLOCK_SIZE>>>(dArray);   // running thread
	cudaThreadSynchronize();
	//epilogue();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapseTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	epilogue();
	printf("GPU time: %f ms\n", gpu_elapseTime);
	
    return 0;
}

// Dla N 1000: CPU 0,000008 ms, GPU 0,761984 ms / 0,027136 ms (bez prologu i epilogu)
// Dla N 1000000: CPU 0,007193, GPU 6,650976 / 0,582208
