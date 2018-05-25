#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
	int width;
	int height;
	float *elements;
} Matrix;

#define BLOCK_SIZE 16

clock_t cpu_startTime, cpu_endTime, gpu_startTime, gpu_endTime;
double cpu_elapseTime, gpu_elapseTime;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void checkCUDAError(const char *msg)
{
	 cudaError_t err = cudaGetLastError();
	 if( cudaSuccess != err) 
	 {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	 }
}

void MatMulCPU(const Matrix A, const Matrix B, Matrix C)
{
	// M(row, col) = *(M.elements + row * M.width + col)
	int i, j, k;
	for (i = 0; i < A.height; i++)
	{
		for (j = 0; j < B.width; j++)
		{
			*(C.elements + i * C.width + j) = 0;
			for (k = 0; k < A.width; k++)
			{
				*(C.elements + i * C.width + j) += 
				(*(A.elements + i * A.width + k)) * (*(B.elements + k * B.width + j));
			}
		}
	}
}

void MatMulGPU(Matrix A, Matrix B, Matrix C)
{
	Matrix d_A;
	d_A.width  = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void **)&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width  = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void **)&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,cudaMemcpyHostToDevice);
	
	Matrix d_C;
	d_C.width  = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void **)&d_C.elements, size);
	
	for (i = 0; i < A.height * A.width; i++)
	{
		A.elements[i] = (float)rand()/(float)(RAND_MAX);
		B.elements[i] = (float)rand()/(float)(RAND_MAX);
	}
	
	gpu_startTime = clock();

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	gpu_endTime = clock();
	gpu_elapseTime = ((gpu_endTime - gpu_startTime)/(double)CLOCKS_PER_SEC);

	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < 10; i++)
	{
		printf("%f ", C.elements[i]);
	}
	printf("\nGPU time: %f ms\n", gpu_elapseTime);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

int mapSupport()
{
	int result = 1;
	cudaDeviceProp deviceProp;
	#if CUDART_VERSION < 2020
	#error "This device does not support memory mapping!\n"
	#endif
	cudaGetDeviceProperties(&deviceProp, 0);
	checkCUDAError("cudaGetDeviceProperties");
	if(!deviceProp.canMapHostMemory) 
	{
		result = 0;
	}
	return result;
}

void MatMulGPUMap(Matrix A, Matrix B, Matrix C)
{
	if (!mapSupport()) 
		printf("This device does not support memory mapping!\n");
	
	Matrix d_A, d_B, d_C;
	int i = 0;
	
	cudaSetDeviceFlags(cudaDeviceMapHost);
	checkCUDAError("cudaSetDeviceFlags");

	d_A.width  = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaHostAlloc((void**)&A.elements, size, cudaHostAllocMapped);
	checkCUDAError("cudaHostAllocMapped");
	cudaHostGetDevicePointer((void**)&d_A.elements, (void*)A.elements, 0);
	checkCUDAError("cudaHostGetDevicePointer");

	d_B.width  = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaHostAlloc((void**)&B.elements, size, cudaHostAllocMapped);
	checkCUDAError("cudaHostAllocMapped");
	cudaHostGetDevicePointer((void**)&d_B.elements, (void*)B.elements, 0);
	checkCUDAError("cudaHostGetDevicePointer");

	d_C.width  = C.width;
	d_C.height = C.height;
	cudaHostAlloc((void**)&C.elements, size, cudaHostAllocMapped);
	checkCUDAError("cudaHostAllocMapped");
	cudaHostGetDevicePointer((void**)&d_C.elements, (void*)C.elements, 0);
	checkCUDAError("cudaHostGetDevicePointer");
	
	for (i = 0; i < A.height * A.width; i++)
	{
		A.elements[i] = (float)rand()/(float)(RAND_MAX);
		B.elements[i] = (float)rand()/(float)(RAND_MAX);
		C.elements[i] = 0;
	}
	
	gpu_startTime = clock();

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	checkCUDAError("MatMulKernel");
	
	cudaThreadSynchronize();
	checkCUDAError("cudaThreadSynchronize");
	
	gpu_endTime = clock();
	gpu_elapseTime = ((gpu_endTime - gpu_startTime)/(double)CLOCKS_PER_SEC);
	
	for (i = 0; i < 10; i++)
	{
		printf("%f ", C.elements[i]);
	}
	printf("\nGPU time: %f ms\n", gpu_elapseTime);

	cudaFreeHost(A.elements);
	cudaFreeHost(B.elements);
	cudaFreeHost(C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char** argv)
{
	if (argc != 2)
		printf("Not enough or too much arguments!");
	
	srand((unsigned int)time(NULL));
	int i = 0;
	
	Matrix A;
	A.width = 1600;
	A.height = 1600;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	
	Matrix B;
	B.width = 1600;
	B.height = 1600;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	
	Matrix C;
	C.width = B.width;
	C.height = A.height;
	C.elements = (float*)malloc(C.height * C.width * sizeof(float));
	
	Matrix D;
	D.width = B.width;
	D.height = A.height;
	D.elements = (float*)malloc(D.height * D.width * sizeof(float));
	
	//GPU:
	if (argv[1] == "--copy")
		MatMulGPU(A, B, D);
	else if (argv[1] == "--map")
		MatMulGPUMap(A, B, D);
	else if (argv[1] == "--auto")
	{
		if(mapSupport())
			MatMulGPUMap(A, B, D);
		else
			MatMulGPU(A, B, D);
	}
	else
		printf("Invalid argument!");
	
	//CPU:
	cpu_startTime = clock();
	MatMulCPU(A, B, C);
	cpu_endTime = clock();
	cpu_elapseTime = ((cpu_endTime - cpu_startTime)/(double)CLOCKS_PER_SEC);
	for (i = 0; i < 10; i++)
	{
		printf("%f ", C.elements[i]);
	}
	printf("\nCPU time: %f ms\n", cpu_elapseTime);
	
	free(A.elements);
	free(B.elements);
	free(C.elements);
	free(D.elements);
	
	return 0;
}