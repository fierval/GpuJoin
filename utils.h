#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// useful utility copy
template<typename inout>
inline void copy(inout * const start, int size, thrust::device_vector<inout>& dest)
{
    thrust::copy(start, start + size, dest.begin());
}

template<typename inout>
inline void copyToHost(const thrust::device_vector<inout>& start, std::vector<inout>& dest)
{
    thrust::copy(start.begin(), start.end(), dest.begin());
}

#endif
