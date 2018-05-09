#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	int width;
	int height;
	float *elements;
} Matrix;

#define BLOCK_SIZE 2

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
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

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
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
	int i = 0;
	
	Matrix A;
	A.width = 2;
	A.height = 4;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	A.elements[0] = 9;
	A.elements[1] = 6;
	A.elements[2] = 3;
	A.elements[3] = 7;
	A.elements[4] = 2;
	A.elements[5] = 1;
	A.elements[6] = 4;
	A.elements[7] = 5;
	
	Matrix B;
	B.width = 4;
	B.height = 2;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	B.elements[0] = 1;
	B.elements[1] = 4;
	B.elements[2] = 6;
	B.elements[3] = 8;
	B.elements[4] = 2;
	B.elements[5] = 4;
	B.elements[6] = 1;
	B.elements[7] = 3;
	
	Matrix C;
	C.width = B.width;
	C.height = A.height;
	C.elements = (float*)malloc(C.height * C.width * sizeof(float));
	
	MatMul(A, B, C);
	for (i; i < C.height * C.width; i++)
	{
		printf("%f ", C[i]);
	}
	
	free(A);
	free(B);
	free(C);
	return 0;
}