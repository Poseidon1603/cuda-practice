#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <new>
#include <stdexcept>
#include <type_traits>

template <typename T> class cudaVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "cudaVector requires trivially copyable T for device transfer");

public:
  // Constructor
  cudaVector() : buf(nullptr), sz(0), cap(0) {}

  // Returns a vector of size n;
  explicit cudaVector(size_t n) : buf(nullptr), sz(0), cap(0) {
    cudaError_t err = cudaMallocHost(&buf, n * sizeof(T));
    if (err != cudaSuccess)
      throw std::runtime_error("cudaMallocHost Failed");
    sz = n;
    cap = n;
  }

  cudaVector(cudaVector &&o) noexcept : buf(o.buf), sz(o.sz), cap(o.cap) {
    o.buf = nullptr;
    o.sz = 0;
    o.cap = 0;
  }

  cudaVector &operator=(cudaVector &&o) noexcept {
    if (this != &o) {
      cudaFreeHost(buf);
      buf = o.buf;
      sz = o.sz;
      cap = o.cap;
      o.buf = nullptr;
      o.sz = 0;
      o.cap = 0;
    }
    return *this;
  }

  // Destructor
  ~cudaVector() { cudaFreeHost(buf); }

  // Allows for operator[] (this is an immutable version)
  const T &operator[](size_t i) const { return buf[i]; }

  // Mutable version of operator[]
  T &operator[](size_t i) { return buf[i]; }

  cudaVector(const cudaVector &) = delete;

  cudaVector &operator=(const cudaVector &) = delete;

  size_t size() const { return sz; }

  size_t capacity() const { return cap; }

  const T *data() const { return buf; }

  T *data() { return buf; }

  T *begin() { return buf; }
  T *end() { return buf + sz; }
  const T *begin() const { return buf; }
  const T *end() const { return buf + sz; }

  void resize(size_t n) {
    // if sizes are the same we dont care
    if (sz == n)
      return;

    if (n > sz) {
      if (n > cap) {
        grow(n, sz);
      }

      // Now we fill the values from previous size to new size
      for (size_t i = sz; i < n; i++) {
        new (&buf[i]) T{};
      }

      sz = n;
    } else if (n < sz) {
      // Call the destructor
      for (size_t i = n; i < sz; i++) {
        buf[i].~T();
      }

      sz = n;
    }
  }

  // N must be greater than current capacity
  void reserve(size_t n) {
    if (cap >= n)
      return;

    grow(n, sz);
  }

  void push_back(T value) {
    if (sz == cap) {
      // define a temp pointer
      size_t tmp_cap = cap ? cap * 2 : 1;

      grow(tmp_cap, cap);
    }

    new (&buf[sz]) T(value);

    sz++;
  }

private:
  // Pointer to list
  T *buf;

  // Returns the number of filed elements
  size_t sz;

  // Returns the number of max possible elements according to the memory
  // allocated
  size_t cap;

  void grow(size_t n, size_t copy_size) {
    if (copy_size > n)
      throw std::runtime_error("Copy Size cant be larger than array size");

    T *tmp = nullptr;

    // Give data the new malloc
    cudaError_t err = cudaMallocHost(&tmp, sizeof(T) * (n));

    if (err != cudaSuccess)
      throw std::runtime_error("cudaMallocHost failed");

    // Copy existing data into the new memory
    if (copy_size)
      cudaMemcpy(tmp, buf, sizeof(T) * copy_size, cudaMemcpyHostToHost);

    // Free tmp / previous memory
    cudaFreeHost(buf);
    buf = tmp;
    cap = n;
  }
};