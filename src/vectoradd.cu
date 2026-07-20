#include <cassert>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <type_traits>

using namespace std;

template <typename T>
__global__ void VectorAdd(T* a, T* b, T* dest, int size) {
    // This gives the global thread we are on 
    // gridDim.x gives the number of blocks per grid
    // blockDim.x gives the number of threads per block
    // blockIdx.x gives the index of the block we are on
    // ThreadIdx.x gives the thread in the block that we are on
    // int index = blockDim.x * blockIdx.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x gives you the max number of threads you can run at once
    

    // Running just threadIdx.x gives the local thread that we are on inside the block
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i+= stride) {
        dest[i] = a[i] + b[i];
    }

    return; 
}

// Using the kernel (assumes a.size() == b.size())
template <typename T>
void VectorAddSetup(vector<T>& a, vector<T>& b, vector<T>& c) {

    T *a_data, *b_data, *c_data;

    c.resize(a.size());

    // Allocate VRAM for a
    cudaMalloc(&a_data, sizeof(T) * a.size());

    // Allocate VRAM for b
    cudaMalloc(&b_data, sizeof(T) * b.size());

    // Allocate VRAM for c
    cudaError_t a_error = cudaMalloc(&c_data, sizeof(T) * c.size());

    // copy into VRAM
    cudaError_t b_error = cudaMemcpy(a_data, a.data(), sizeof(T) * a.size(), cudaMemcpyHostToDevice);

    cudaError_t c_error = cudaMemcpy(b_data, b.data(), sizeof(T) * b.size(), cudaMemcpyHostToDevice);

    // Get number of streaming multiprocessors
    cudaDeviceProp prop;
    int deviceID = 0;
    cudaGetDeviceProperties(&prop, deviceID);
    int numSMs = prop.multiProcessorCount;

    // Calculate optimal number of blocks (per sm) to use at 256 threads
    int maxBlockPerSM = 0;
    int threadsPerBlock = 256;
    size_t dynamicSMemSize = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, VectorAdd<T>, threadsPerBlock, dynamicSMemSize);
    
    VectorAdd<<<numSMs * maxBlockPerSM, threadsPerBlock>>>(a_data, b_data, c_data, a.size());

    // Copy back to host
    cudaMemcpy(c.data(), c_data, sizeof(T) * c.size(), cudaMemcpyDeviceToHost);

    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(c_data);
}

// CPU only implementation
template <typename T>
requires (!((is_same_v<T, string>) || (is_same_v<T, char>)))
vector<T> VectorAddCPU(vector<T>& a, vector<T>& b) {
    int size = a.size() >= b.size() ? a.size() : b.size();
    vector<T> result(size);
    for (int i = 0; i < size; i++) {
        if (i > a.size()) {
            a.emplace_back();
        }
        if (i > b.size()) {
            b.emplace_back();
        }

        result[i] = a[i] + b[i];
    }

    return result;
}

int main() {
    vector<int> a = {1,2,3};
    vector<int> b = {2,3,4};

    vector<int> c = VectorAddCPU(a, b);

    for (const auto& num : c) {
        cout << num; 
        if (&num != &c.back()) {
            cout << ",";
        }
    }

    cout << "\n";
    
    int size = 0;
    cout << "Size of vector to add? \n";
    cin >> size;
    vector<int> d(size);
    
    iota(d.begin(), d.end(), 1);

    vector<int> e(size);

    iota(e.begin(), e.end(), 1);

    vector<int> f;

    VectorAddSetup<int>(d, e,f);

    for (const auto& num : f) { 
        cout << num;
        if (&num != &f.back()) {
            cout << ",";
        }
    }

    cout << "\n";

    return 0;
}
