#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void simpleKernel() {
	printf("Hello World \n");
}

int main() {
	simpleKernel<<<1,1>>>();

	cudaDeviceSynchronize();

	return 0;
}
