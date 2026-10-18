#pragma once

#include <stdlib.h>
#include <iostream>
#include <algorithm>

template <typename T>
std::pair<int, int> gridSize(T kernel, size_t dynamicSMemSize = 0,
                             size_t threadsPerBlock = 256) {
  cudaDeviceProp prop;
  int deviceID = 0;
  cudaGetDeviceProperties(&prop, deviceID);
  int numSMs = prop.multiProcessorCount;

  // Calculate optimal number of blocks (per sm) to use at 256 threads
  int maxBlockPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlockPerSM, kernel, threadsPerBlock, dynamicSMemSize);
  return {numSMs * maxBlockPerSM, threadsPerBlock};
};

template <typename T>
void OutStats(std::vector<T> &samples, size_t vectorsize, size_t typesize,
              size_t arraystouched) {
  if (samples.empty()) {
    std::cout << "Samples are empty" << "\n";
  }
  float total = 0;

  for (const auto &sample : samples) {
    // std::cout << sample << "\n";
    total += sample;
  }

  std::cout << "Mean: " << total / samples.size() << "ms" << "\n";

  // Sorted for median
  std::sort(samples.begin(), samples.end());
  size_t m = samples.size();
  float median = m % 2 == 0 ? (samples[m / 2 - 1] + samples[m / 2]) / 2.0f
                            : samples[m / 2];

  std::cout << "Median: " << median << "ms" << "\n";
  std::cout << "Min: " << samples.front() << "ms" << "\n";
  std::cout << "Top 5%: " << samples[round(samples.size() * 0.05)] << "ms"
            << "\n";
  std::cout << "Top 25%: " << samples[round(samples.size() * 0.25)] << "ms"
            << "\n";
  std::cout << "Bottom 5%: " << samples[round(samples.size() * 0.95)] << "ms"
            << "\n";
  std::cout << "Bottom 25%: " << samples[round(samples.size() * 0.75)] << "ms"
            << "\n";

  std::cout << "BANDWIDTH (gb/s) \n";
  std::cout << "Median Bandwidth: "
            << (arraystouched * vectorsize * typesize) / 1e9 / (median / 1000)
            << "\n";
  std::cout << "Min (Runtime) Bandwidth: "
            << (arraystouched * vectorsize * typesize) / 1e9 /
                   (samples.front() / 1000)
            << "\n";
}