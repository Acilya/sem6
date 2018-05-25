#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

void MatMulCPU(Matrix A, Matrix B)
{
	Matrix D;
	D.width = B.width;
	D.height = A.height;
	D.elements = (float*)malloc(D.height * D.width * sizeof(float));
		
	cpu_startTime = clock();
	
	// M(row, col) = *(M.elements + row * M.width + col)
	int i, j, k;
	for (i = 0; i < A.height; i++)
	{
		for (j = 0; j < B.width; j++)
		{
			*(D.elements + i * D.width + j) = 0;
			for (k = 0; k < A.width; k++)
			{
				*(D.elements + i * D.width + j) += 
				(*(A.elements + i * A.width + k)) * (*(B.elements + k * B.width + j));
			}
		}
	}
	
	cpu_endTime = clock();
	cpu_elapseTime = ((cpu_endTime - cpu_startTime)/(double)CLOCKS_PER_SEC);
	for (i = 0; i < 10; i++)
	{
		printf("%f ", D.elements[i]);
	}
	printf("\nCPU time: %f ms\n", cpu_elapseTime);
	
	free(D.elements);
}

void MatMulGPU(Matrix A, Matrix B, Matrix C)
{
	int i = 0;
	for (i = 0; i < A.height * A.width; i++)
	{
		A.elements[i] = (float)rand()/(float)(RAND_MAX);
		B.elements[i] = (float)rand()/(float)(RAND_MAX);
	}
	
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
	
	gpu_startTime = clock();

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	gpu_endTime = clock();
	gpu_elapseTime = ((gpu_endTime - gpu_startTime)/(double)CLOCKS_PER_SEC);

	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	
	for (i = 0; i < 10; i++)
	{
		printf("%f ", C.elements[i]);
	}
	printf("\nGPU time: %f ms\n", gpu_elapseTime);
	
	MatMulCPU(A, B);
}

int checkMapSupport()
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
	if (!checkMapSupport()) 
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
	
	MatMulCPU(A, B);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
	{
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	}
	C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Not enough or too much arguments!\n");
	}
	else
	{
		srand((unsigned int)time(NULL));

		Matrix A;
		A.width = 16;
		A.height = 16;
		A.elements = (float*)malloc(A.width * A.height * sizeof(float));

		Matrix B;
		B.width = 16;
		B.height = 16;
		B.elements = (float*)malloc(B.width * B.height * sizeof(float));

		Matrix C;
		C.width = B.width;
		C.height = A.height;
		C.elements = (float*)malloc(C.height * C.width * sizeof(float));

		//GPU:
		if (strcmp(argv[1], "--copy") == 0)
			MatMulGPU(A, B, C);
		else if (strcmp(argv[1], "--map") == 0)
			MatMulGPUMap(A, B, C);
		else if (strcmp(argv[1], "--auto") == 0)
		{
			if(checkMapSupport())
				MatMulGPUMap(A, B, C);
			else
				MatMulGPU(A, B, C);
		}
		else
			printf("Invalid argument!");

		free(A.elements);
		free(B.elements);
		free(C.elements);
	}
	return 0;
}