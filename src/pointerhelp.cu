#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

__global__ void simpleAccessKernel(int& a) {}

__global__ void simpleSquaredKernel(int&  a) {
	 a *= a;
}

struct Example {
	int id;
	int num;
	char letter;
};


int main() {
	std::ios_base::sync_with_stdio(false);
	cin.tie(NULL);

	// Classic pointer manipulation
	int a = 5;

	printf("Value of A: %d \n", a);

	int* b = &a;

	printf("Pointer B: %p \n", b);
	printf("Mem Location of A: %p \n", &a);

	*b = 10;

	printf("Value of A again: %d \n", a);

	// Malloc allocations work by putting memory on the heap
	// Got to remember to cast the pointer to an int*
	int *p = (int *)malloc(sizeof(int));

	// Now we have an int pointer on the heap
	printf("Pointer P: %p \n", p);

	// Dereferencing the pointer allows us to modify the data
	*p = 10;

	printf("Value of P: %d \n", *p);

	free(p);
	
	int n;

	printf("Enter Number of Elements: \n");

	cin >> n;

	// Heap Allocation of a list
	int *arr = (int *)malloc(n * sizeof(int));
	
	for (int i = 0; i < n; i++) {
		arr[i] = i + 1;
	}

	for (int i = 0; i < n; i++) {
		printf("Value: %d \n", arr[i]);
	}

	free(arr);

	// Allocating a struct on the heap
	
	struct Example* eg = (struct Example *)malloc(sizeof(struct Example));

	if (eg == NULL) {
		printf("Mem Alloc Failed \n");
		return 1;
	}

	eg->id = 0;
	eg->letter = 'a';
	eg->num = 128;
	
	printf("Example Struct Vals: \n");
	printf("	ID: %d \n", eg->id);
	printf("	NUM: %d \n", eg->num);
	printf("	LETTER: %c \n", eg->letter);
	
	free(eg);

	// Allocation a 2D Array on the heap (using 1D math)
	int c;
	int r;

	printf("How Many Top Level Elements? \n");
	cin >> c;

	printf("How Many Sub Elements? \n");
	cin >> r;

	int* arr2 = (int *)malloc(c * r * sizeof(int));

	for (int i = 0; i < (r * c); i++) {
		arr2[i] = i * 10 + 1;
	}

	// Accessing requires knowing the # of collumns and the # of rows
	for (int i = 0; i < (r * c); i++) {
		printf("Value: %d \n", arr2[i]);
	}
	
	free(arr2);

	// 2D Alloc using array of pointers
	int cols;
	int rows;

	printf("Number of Collumns: ");
	fflush(stdout);
	cin >> cols;

	printf("Number of Rows: ");
	fflush(stdout);
	cin >> rows;

	int** arr3 = (int **)malloc(cols * sizeof(int*));

	for (int i = 0; i < cols; i++) {
		arr3[i] = (int *)malloc(rows * sizeof(int));
	}

	// Fill the Array
	for (int i = 0; i < cols; i++) {
		for (int a = 0; a < rows; a++) {
			arr3[i][a] = i + 1;
		}
	}

	// Print Out
	for (int i = 0; i < cols; i++) {
		string out;
		for (int a = 0; a < rows; a++) {
			out += to_string(arr3[i][a]);
		}

		printf("[%s]," ,out.c_str());
	}

	printf("\n");

	// Free up the arrays

	for (int i = 0; i < cols; i++) {
		free(arr3[i]);
	}

	free(arr3);

	// VRAM Malloc

	// First Malloc into RAM
	int* h_data = (int*)malloc(sizeof(int));

	// Malloc the same size into VRAM
	int* d_data;
	cudaMalloc(&d_data, sizeof(int));

	// Assign data to RAM
	*h_data = 10;

	// Copy Data From RAM to VRAM
	cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);

	simpleSquaredKernel<<<1,1>>>(*d_data);

	// Copy back from device to host (VRAM to RAM)

	cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Returned Value: %d \n", *h_data);

	cudaFree(d_data);


	// VRAM Usage through unified memory

	int* d_h_data;
	
	cudaMallocManaged(&d_h_data, sizeof(int));

	*d_h_data = 10;
	
	simpleSquaredKernel<<<1,1>>>(*d_h_data);

	cudaDeviceSynchronize();

	printf("Unified Mem Value: %d \n", *d_h_data);

	cudaFree(d_h_data);

	// cudaMallocManaged can introduce memory overhead because its moving the data 
	// between RAM and VRAM in the background on demand, to combat this you could pprefetch the data

	int* d_h_data2;
	int deviceId;
	
	cudaGetDevice(&deviceId);
	
	cudaMallocManaged(&d_h_data2, sizeof(int));

	*d_h_data2 = 10;

	cudaMemPrefetchAsync(&d_h_data2, sizeof(int), deviceId);

	simpleSquaredKernel<<<1,1>>>(*d_h_data2);

	cudaMemPrefetchAsync(&d_h_data2, sizeof(int), cudaCpuDeviceId);

	cudaDeviceSynchronize();
	
	printf("Value Returned: %d  \n", *d_h_data2);

	cudaFree(d_h_data2);

	return 0;
}


