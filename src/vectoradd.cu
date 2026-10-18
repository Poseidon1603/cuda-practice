#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>

#include "cudaVector.cuh"
#include "lib.cuh"

template <typename T> __global__ void VectorAdd(T *a, T *b, T *dest, int size) {
  // This gives the global thread we are on
  // gridDim.x gives the number of blocks per grid
  // blockDim.x gives the number of threads per block
  // blockIdx.x gives the index of the block we are on
  // ThreadIdx.x gives the thread in the block that we are on
  // int index = blockDim.x * blockIdx.x + threadIdx.x;
  // int stride = blockDim.x * gridDim.x gives you the max number of threads you
  // can run at once

  // Running just threadIdx.x gives the local thread that we are on inside the
  // block
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < size; i += stride) {
    dest[i] = a[i] + b[i];
  }

  return;
}

// Using the kernel (assumes a.size() == b.size())
template <typename T>
void VectorAddSetup(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c) {
  c.resize(a.size());

  // Get our data ready
  T *a_data, *b_data, *c_data;
  cudaMalloc(&a_data, sizeof(T) * a.size());
  cudaMalloc(&b_data, sizeof(T) * b.size());
  cudaError_t c_error = cudaMalloc(&c_data, sizeof(T) * c.size());
  cudaEvent_t start, stop, h2d_start, h2d_stop, d2h_start, d2h_stop, k_start,
      k_stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&h2d_start);
  cudaEventCreate(&h2d_stop);
  cudaEventCreate(&d2h_start);
  cudaEventCreate(&d2h_stop);
  cudaEventCreate(&k_start);
  cudaEventCreate(&k_stop);

  auto [grid, threads] = gridSize(VectorAdd<T>);

  // Warmup runs
  for (int i = 0; i < 100; i++) {
    cudaError_t a_error = cudaMemcpy(a_data, a.data(), sizeof(T) * a.size(),
                                     cudaMemcpyHostToDevice);
    cudaError_t b_error = cudaMemcpy(b_data, b.data(), sizeof(T) * b.size(),
                                     cudaMemcpyHostToDevice);

    if (a_error != cudaSuccess || b_error != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }

    VectorAdd<<<grid, threads>>>(a_data, b_data, c_data, a.size());

    cudaError_t c_out = cudaMemcpy(c.data(), c_data, sizeof(T) * c.size(),
                                   cudaMemcpyDeviceToHost);

    if (c_out != cudaSuccess)
      throw std::runtime_error("Failed to copy back to host");
  }

  cudaDeviceSynchronize();

  // Confirm correctnes
  for (size_t i = 0; i < c.size(); i++) {
    if (c[i] != (a[i] + b[i])) {
      std::cout << c[i] << "\n";
      std::cout << b[i] << "\n";
      std::cout << a[i] << "\n";
      throw std::runtime_error("Addition is incorrect");
    }
  }

  std::vector<float> samples, h2d_samples, d2h_samples, k_samples;
  for (int i = 0; i < 1000; ++i) {
    // Record overall time taken
    cudaEventRecord(start);

    // Record Time Taken to H2D
    cudaEventRecord(h2d_start);
    cudaError_t a_err = cudaMemcpy(a_data, a.data(), sizeof(T) * a.size(),
                                   cudaMemcpyHostToDevice);
    cudaError_t b_err = cudaMemcpy(b_data, b.data(), sizeof(T) * b.size(),
                                   cudaMemcpyHostToDevice);

    if (!(a_err == cudaSuccess && b_err == cudaSuccess)) {
      throw std::runtime_error("Failed to cudaMemcpy to device");
    }
    cudaEventRecord(h2d_stop);

    // Record Kernel Time Taken
    cudaEventRecord(k_start);
    VectorAdd<<<grid, threads>>>(a_data, b_data, c_data, a.size());
    cudaEventRecord(k_stop);

    // Record Time Take for D2H
    cudaEventRecord(d2h_start);
    cudaError_t c_err = cudaMemcpy(c.data(), c_data, sizeof(T) * c.size(),
                                   cudaMemcpyDeviceToHost);

    if (!(c_err == cudaSuccess)) {
      throw std::runtime_error("Failed to cudaMemcpy back to host");
    }
    cudaEventRecord(d2h_stop);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, h2d_start, h2d_stop);
    h2d_samples.push_back(ms);
    cudaEventElapsedTime(&ms, k_start, k_stop);
    k_samples.push_back(ms);
    cudaEventElapsedTime(&ms, d2h_start, d2h_stop);
    d2h_samples.push_back(ms);
    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);
  }

  std::cout << "Overall Time Samples" << "\n";
  OutStats(samples, a.size(), sizeof(T), 3);
  std::cout << "\n";

  std::cout << "H2D Samples" << "\n";
  OutStats(h2d_samples, a.size(), sizeof(T), 2);
  std::cout << "\n";

  std::cout << "D2H Samples" << "\n";
  OutStats(d2h_samples, a.size(), sizeof(T), 1);
  std::cout << "\n";

  std::cout << "Kernel Samples" << "\n";
  OutStats(k_samples, a.size(), sizeof(T), 3);
  std::cout << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_stop);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_stop);
  cudaEventDestroy(k_start);
  cudaEventDestroy(k_stop);
  cudaFree(a_data);
  cudaFree(b_data);
  cudaFree(c_data);
}

// cudaVector with proper metric implementation
//  This assumes vectors are the same size
template <typename T>
void cudaVectorAdd(cudaVector<T> &a, cudaVector<T> &b, cudaVector<T> &dest) {
  dest.resize(a.size());

  // Event Recording
  cudaEvent_t start, stop, h2d_start, h2d_stop, d2h_start, d2h_stop, k_start,
      k_stop;
  std::vector<float> samples, h2d_samples, d2h_samples, k_samples;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&h2d_start);
  cudaEventCreate(&h2d_stop);
  cudaEventCreate(&d2h_start);
  cudaEventCreate(&d2h_stop);
  cudaEventCreate(&k_start);
  cudaEventCreate(&k_stop);

  T *a_data, *b_data, *c_data;
  cudaMalloc(&a_data, sizeof(T) * a.size());
  cudaMalloc(&b_data, sizeof(T) * b.size());
  cudaMalloc(&c_data, sizeof(T) * dest.size());

  auto [grid, threads] = gridSize(VectorAdd<T>);

  // warm up runs
  for (size_t i = 0; i < 100; i++) {
    cudaMemcpy(a_data, a.data(), sizeof(T) * a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, b.data(), sizeof(T) * b.size(), cudaMemcpyHostToDevice);

    VectorAdd<<<grid, threads>>>(a_data, b_data, c_data, a.size());

    cudaMemcpy(dest.data(), c_data, sizeof(T) * dest.size(),
               cudaMemcpyDeviceToHost);
  }

  // Measured Runs
  for (size_t i = 0; i < 1000; i++) {
    // Overall recording
    cudaEventRecord(start);

    // Memcpy recording
    cudaEventRecord(h2d_start);
    cudaError_t a_err = cudaMemcpy(a_data, a.data(), sizeof(T) * a.size(),
                                   cudaMemcpyHostToDevice);
    cudaError_t b_err = cudaMemcpy(b_data, b.data(), sizeof(T) * b.size(),
                                   cudaMemcpyHostToDevice);

    if (!(a_err == cudaSuccess && b_err == cudaSuccess)) {
      throw std::runtime_error("Failed to cudaMemcpy");
    }
    cudaEventRecord(h2d_stop);

    // Kernel recording
    cudaEventRecord(k_start);
    VectorAdd<<<grid, threads>>>(a_data, b_data, c_data, a.size());
    cudaEventRecord(k_stop);

    // Memcpy to Host recording
    cudaEventRecord(d2h_start);
    cudaError_t c_err = cudaMemcpy(dest.data(), c_data, sizeof(T) * dest.size(),
                                   cudaMemcpyDeviceToHost);

    if (c_err != cudaSuccess) {
      throw std::runtime_error("Failed to Memcpy back to host");
    }
    cudaEventRecord(d2h_stop);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);
    cudaEventElapsedTime(&ms, h2d_start, h2d_stop);
    h2d_samples.push_back(ms);
    cudaEventElapsedTime(&ms, k_start, k_stop);
    k_samples.push_back(ms);
    cudaEventElapsedTime(&ms, d2h_start, d2h_stop);
    d2h_samples.push_back(ms);
  }

  std::cout << "Overall Time \n";
  OutStats(samples, a.size(), sizeof(T), 3);

  std::cout << "H2D Samples \n";
  OutStats(h2d_samples, a.size(), sizeof(T), 2);

  std::cout << "Kernel Samples \n";
  OutStats(k_samples, a.size(), sizeof(T), 3);

  std::cout << "D2H Samples \n";
  OutStats(d2h_samples, a.size(), sizeof(T), 1);

  cudaFree(a_data);
  cudaFree(b_data);
  cudaFree(c_data);
}

// With streams we can chunk the data up
template <typename T>
void cudaVectorAddStreams(cudaVector<T> &a, cudaVector<T> &b,
                          cudaVector<T> &dest, size_t stream_num) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> samples, h2d_samples, d2h_samples, k_samples;

  dest.resize(a.size());
  T *a_data, *b_data, *c_data;

  cudaMalloc(&a_data, sizeof(T) * a.size());
  cudaMalloc(&b_data, sizeof(T) * b.size());
  cudaMalloc(&c_data, sizeof(T) * dest.size());

  // List of streams
  std::vector<cudaStream_t> streams(stream_num);
  for (auto &stream : streams) {
    cudaStreamCreate(&stream);
  }

  auto [grid, threads] = gridSize(VectorAdd<T>);

  // warm up
  size_t chunkSize = (a.size() + stream_num - 1) / stream_num;

  cudaMemcpy(a_data, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(b_data, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);

  for (size_t j = 0; j < 5; j++) {
    for (size_t i = 0; i < stream_num; i++) {
      size_t offset = i * chunkSize;
      size_t elemCount = std::min(chunkSize, a.size() - offset);
      VectorAdd<<<grid, threads, 0, streams[i]>>>(
          a_data + offset, b_data + offset, c_data + offset, elemCount);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(dest.data(), c_data, dest.size() * sizeof(T),
               cudaMemcpyDeviceToHost);

    for (size_t h = 0; h < dest.size(); h++) {
      if (dest[h] != a[h] + b[h]) {
        std::cout << "Values Given: \n";
        std::cout << "Dest: " << dest[h] << "\n";
        std::cout << "Vector 1: " << a[h] << "\n";
        std::cout << "Vector 2: " << b[h] << "\n";
        std::cout << "Element ID: " << h << "\n";

        throw std::runtime_error("Stream Addition incorrect");
      }
    }
  }

  float ms;
  // Measured Runs
  for (size_t runs = 0; runs < 1000; runs++) {
    cudaEventRecord(start);

    for (size_t i = 0; i < stream_num; i++) {
      size_t offset = i * chunkSize;
      size_t elemCount = std::min(chunkSize, a.size() - offset);
      size_t bytes = elemCount * sizeof(T);

      // Async Mem Copy
      cudaError_t a_copy =
          cudaMemcpyAsync(a_data + offset, a.data() + offset, bytes,
                          cudaMemcpyHostToDevice, streams[i]);
      cudaError_t b_copy =
          cudaMemcpyAsync(b_data + offset, b.data() + offset, bytes,
                          cudaMemcpyHostToDevice, streams[i]);
      if (!(a_copy == cudaSuccess && b_copy == cudaSuccess)) {
        throw std::runtime_error("Failed to copy");
      }

      // Kernel
      VectorAdd<<<grid, threads, 0, streams[i]>>>(
          a_data + offset, b_data + offset, c_data + offset, elemCount);

      // Async Mem Copy
      cudaError_t c_copy =
          cudaMemcpyAsync(dest.data() + offset, c_data + offset, bytes,
                          cudaMemcpyDeviceToHost, streams[i]);
      if (c_copy != cudaSuccess) {
        throw std::runtime_error("Failed to copy to Host");
      }
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);
  }

  OutStats(samples, a.size(), sizeof(T), 3);

  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a_data);
  cudaFree(b_data);
  cudaFree(c_data);
}

// CPU only implementation
template <typename T>
  requires(!((std::is_same_v<T, std::string>) || (std::is_same_v<T, char>)))
std::vector<T> VectorAddCPU(std::vector<T> &a, std::vector<T> &b) {
  std::vector<T> result(a.size());
  std::vector<double> samples;

  // 1000 runs
  for (size_t j = 0; j < 1000; j++) {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t k = 0; k < a.size(); k++) {
      result[k] = a[k] + b[k];
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0);
    samples.push_back(ms.count());
  }
  volatile T sink = result[result.size() / 2];

  OutStats(samples, a.size(), sizeof(T), 3);
  return result;
}

int main() {
  cudaFree(0);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "asyncEngineCount: " << prop.asyncEngineCount << "\n";

  int size = 0;
  std::cout << "Size of vector to add? \n";
  std::cin >> size;

  std::vector<int> a(size);
  std::vector<int> b(size);

  iota(a.begin(), a.end(), 1);

  iota(b.begin(), b.end(), 1);

  std::cout << "CPU Version \n";

  std::vector<int> c = VectorAddCPU(a, b);

  std::vector<int> d(size);

  iota(d.begin(), d.end(), 1);

  std::vector<int> e(size);

  iota(e.begin(), e.end(), 1);

  std::vector<int> f;

  VectorAddSetup(d, e, f);

  // Now with my cudaVector implementation
  cudaVector<int> cVa(size);

  std::iota(cVa.begin(), cVa.end(), 1);

  cudaVector<int> cVb(size);

  std::iota(cVb.begin(), cVb.end(), 1);

  cudaVector<int> cVc;

  std::cout << "Pinned Page-Locked Version \n";

  cudaVectorAdd(cVa, cVb, cVc);

  std::cout << "Pinned Page-Locked Version + Streams \n";

  cudaVectorAddStreams(cVa, cVb, cVc, 8);

  return 0;
}
