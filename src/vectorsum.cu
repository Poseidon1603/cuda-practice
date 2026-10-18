#include <chrono>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <driver_types.h>
#include <iostream>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cudaVector.cuh"
#include "lib.cuh"

template <typename T> __global__ void VectorSum(T *data, T *sum, size_t size) {
  T localSum = T{};

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t i = idx; i < size; i += gridDim.x * blockDim.x) {
    localSum += data[i];
  }

  __shared__ T sData[256];
  int tid = threadIdx.x;

  sData[tid] = localSum;
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sData[tid] += sData[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(sum, sData[0]);
}

template <typename T> void VectorSumGPU(std::vector<T> &data) {
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

  std::vector<float> samples, h2d_samples, d2h_samples, k_samples;

  // Device Pointers
  T *d_data, *result;

  // Host var for result
  T sum{};

  // For verification
  T cpu_sum{};
  for (const auto &num : data) {
    cpu_sum += num;
  }

  float ms;
  auto [grid, threads] = gridSize(VectorSum<T>);

  cudaMalloc(&d_data, data.size() * sizeof(T));
  cudaMalloc(&result, sizeof(T));

  // warm up
  for (size_t i = 0; i < 100; i++) {
    cudaError_t d_copy = cudaMemcpy(
        d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (d_copy != cudaSuccess)
      throw std::runtime_error("Failed to Copy to Device");
    cudaMemset(result, 0, sizeof(T));
    VectorSum<<<grid, threads>>>(d_data, result, data.size());
    cudaError_t h_copy =
        cudaMemcpy(&sum, result, sizeof(T), cudaMemcpyDeviceToHost);
    if (h_copy != cudaSuccess) {
      throw std::runtime_error("Failed To Copy to Host");
    }

    if (sum != cpu_sum) {
      throw std::runtime_error("Sum is incorrect");
    }
  }

  // Measured Runs
  for (size_t a = 0; a < 1000; a++) {
    cudaEventRecord(start);

    cudaEventRecord(h2d_start);
    cudaError_t a_copy = cudaMemcpy(
        d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (a_copy != cudaSuccess) {
      throw std::runtime_error("Failed to Copy to Device");
    }
    cudaEventRecord(h2d_stop);

    cudaEventRecord(k_start);
    cudaMemset(result, 0, sizeof(T));
    VectorSum<<<grid, threads>>>(d_data, result, data.size());
    cudaEventRecord(k_stop);

    cudaEventRecord(d2h_start);
    cudaError_t sum_copy =
        cudaMemcpy(&sum, result, sizeof(T), cudaMemcpyDeviceToHost);
    if (sum_copy != cudaSuccess) {
      throw std::runtime_error("Failed to Copy to Host");
    }
    cudaEventRecord(d2h_stop);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);
    cudaEventElapsedTime(&ms, h2d_start, h2d_stop);
    h2d_samples.push_back(ms);
    cudaEventElapsedTime(&ms, k_start, k_stop);
    k_samples.push_back(ms);
    cudaEventElapsedTime(&ms, d2h_start, d2h_stop);
    d2h_samples.push_back(ms);
  }

  if (sum != cpu_sum) {
    throw std::runtime_error("Sum is incorrect");
  }

  std::cout << "End To End Samples \n";
  OutStats(samples, data.size(), sizeof(T), 1);
  std::cout << "H2D Samples \n";
  OutStats(h2d_samples, data.size(), sizeof(T), 1);
  std::cout << "Kernel Samples \n";
  OutStats(k_samples, data.size(), sizeof(T), 1);
  std::cout << "D2H Samples \n";
  OutStats(d2h_samples, 1, sizeof(T), 1);

  cudaFree(d_data);
  cudaFree(result);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_stop);
  cudaEventDestroy(k_start);
  cudaEventDestroy(k_stop);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_stop);
}

template <typename T> void cudaVectorSumGPU(cudaVector<T> &data) {
  cudaEvent_t start, stop, h2d_start, h2d_stop, k_start, k_stop, d2h_start,
      d2h_stop;
  std::vector<float> samples, k_samples, h2d_samples, d2h_samples;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&h2d_start);
  cudaEventCreate(&h2d_stop);
  cudaEventCreate(&k_start);
  cudaEventCreate(&k_stop);
  cudaEventCreate(&d2h_start);
  cudaEventCreate(&d2h_stop);

  // Device Pointers
  T *d_data, *d_result;

  // Host var for result
  T sum{};

  // For Checking correctness
  T cpu_sum{};
  for (const auto &num : data) {
    cpu_sum += num;
  }

  cudaMalloc(&d_data, data.size() * sizeof(T));
  cudaMalloc(&d_result, sizeof(T));
  auto [grid, threads] = gridSize(VectorSum<T>);

  // warm up
  for (size_t i = 0; i < 100; i++) {
    cudaMemset(d_result, 0, sizeof(T));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(T),
               cudaMemcpyHostToDevice);

    VectorSum<<<grid, threads>>>(d_data, d_result, data.size());

    cudaMemcpy(&sum, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    if (sum != cpu_sum) {
      throw std::runtime_error("Sum is incorrect");
    }
  }

  // measured runs
  float ms;
  for (size_t i = 0; i < 1000; i++) {
    cudaMemset(d_result, 0, sizeof(T));

    cudaEventRecord(start);

    cudaEventRecord(h2d_start);
    cudaError_t d_copy = cudaMemcpy(
        d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (d_copy != cudaSuccess) {
      throw std::runtime_error("Failed to Copy to Device");
    }
    cudaEventRecord(h2d_stop);

    cudaEventRecord(k_start);
    VectorSum<<<grid, threads>>>(d_data, d_result, data.size());
    cudaEventRecord(k_stop);

    cudaEventRecord(d2h_start);
    cudaError_t h_copy =
        cudaMemcpy(&sum, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    if (h_copy != cudaSuccess) {
      throw std::runtime_error("Failed to Copy to Host");
    }
    cudaEventRecord(d2h_stop);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);
    cudaEventElapsedTime(&ms, h2d_start, h2d_stop);
    h2d_samples.push_back(ms);
    cudaEventElapsedTime(&ms, k_start, k_stop);
    k_samples.push_back(ms);
    cudaEventElapsedTime(&ms, d2h_start, d2h_stop);
    d2h_samples.push_back(ms);
  }

  if (sum != cpu_sum) {
    throw std::runtime_error("Sum is incorrect");
  }

  std::cout << "\nEnd To End Samples \n\n";

  OutStats(samples, data.size(), sizeof(T), 1);

  std::cout << "\nH2D Samples \n\n";

  OutStats(h2d_samples, data.size(), sizeof(T), 1);

  std::cout << "\nKernel Samples \n\n";

  OutStats(k_samples, data.size(), sizeof(T), 1);

  std::cout << "\nD2H Samples \n\n";

  OutStats(d2h_samples, 1, sizeof(T), 1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_stop);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_stop);
  cudaEventDestroy(k_start);
  cudaEventDestroy(k_stop);

  cudaFree(d_data);
  cudaFree(d_result);
}

template <typename T>
void cudaVectorSumStreams(cudaVector<T> &data, size_t streamNum) {
  cudaEvent_t start, stop;
  std::vector<float> samples;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Device Pointer
  T *d_data, *d_result;

  // Host Pointer
  T sum{};

  // For checking correctness
  T cpu_sum = T{};

  for (const auto &num : data) {
    cpu_sum += num;
  }

  // Chunking for the streams
  size_t chunkSize = (data.size() + streamNum - 1) / streamNum;

  cudaMalloc(&d_data, data.size() * sizeof(T));
  cudaMalloc(&d_result, sizeof(T));
  cudaMemset(d_result, 0, sizeof(T));

  // Thread info
  auto [grid, threads] = gridSize(VectorSum<T>);
  float ms;

  std::vector<cudaStream_t> streams(streamNum);

  for (auto &stream : streams) {
    cudaStreamCreate(&stream);
  }

  // warm up runs
  for (size_t i = 0; i < 100; i++) {
    cudaMemset(d_result, 0, sizeof(T));

    sum = T{};
    for (size_t j = 0; j < streamNum; j++) {
      size_t offset = j * chunkSize;
      size_t elems = std::min(chunkSize, data.size() - offset);
      size_t bytes = elems * sizeof(T);
      cudaMemcpyAsync(d_data + offset, data.data() + offset, bytes,
                      cudaMemcpyHostToDevice, streams[j]);

      VectorSum<<<grid, threads, 0, streams[j]>>>(d_data + offset, d_result,
                                                  elems);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&sum, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    if (sum != cpu_sum) {
      throw std::runtime_error("Sum is incorrect");
    }
  }

  // measured runs
  for (size_t i = 0; i < 1000; i++) {
    cudaMemset(d_result, 0, sizeof(T));
    sum = T{};
    cudaEventRecord(start);
    for (size_t j = 0; j < streamNum; j++) {
      size_t offset = j * chunkSize;
      size_t elems = std::min(chunkSize, data.size() - offset);
      size_t bytes = elems * sizeof(T);

      cudaMemcpyAsync(d_data + offset, data.data() + offset, bytes,
                      cudaMemcpyHostToDevice, streams[j]);

      VectorSum<<<grid, threads, 0, streams[j]>>>(d_data + offset, d_result,
                                                  elems);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms);

    cudaDeviceSynchronize();

    cudaMemcpy(&sum, d_result, sizeof(T), cudaMemcpyDeviceToHost);
  }

  if (sum != cpu_sum) {
    throw std::runtime_error("Sum is incorrect");
  }

  std::cout << "\ncudaVectorStreams End to End\n\n";
  OutStats(samples, data.size(), sizeof(T), 1);

  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_data);
  cudaFree(d_result);
}

template <typename T> void VectorSumCPU(std::vector<T> &data) {
  std::vector<double> samples;

  for (size_t i = 0; i < 100; i++) {
    T total{};
    auto t0 = std::chrono::steady_clock::now();
    for (const auto &a : data) {
      total += a;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0);
    samples.push_back(ms.count());

    volatile T sink = total / 2;
  }

  std::cout << "CPU Samples \n";
  OutStats(samples, data.size(), sizeof(T), 1);
}

int main(void) {
  cudaFree(0);

  std::vector<int> a;

  a.resize(10000000);

  std::iota(a.begin(), a.end(), 1);

  VectorSumCPU(a);

  VectorSumGPU(a);

  cudaVector<int> cVa;

  cVa.resize(10000000);

  std::iota(cVa.begin(), cVa.end(), 1);

  cudaVectorSumGPU(cVa);

  cudaVectorSumStreams(cVa, 8);

  return 0;
}