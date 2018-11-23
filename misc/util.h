#pragma once

#include <cmath>

#ifdef __JETBRAINS_IDE__

#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads(){};
inline void __threadfence_block(){};
inline void __threadfence(){};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate){return predicate};
inline int __syncthreads_and(int predicate){return predicate};
inline int __syncthreads_or(int predicate){return predicate};
template <class T>
inline T __clz(const T val) {
  return val;
}
template <class T>
inline T __ldg(const T *address){return *address};
// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;

typedef struct uchar1 { uchar x; } uchar1;

typedef struct uchar2 {
  uchar x;
  uchar y;
} uchar2;

typedef struct uchar3 {
  uchar x;
  uchar y;
  uchar z;
} uchar3;

typedef struct uchar4 {
  uchar x;
  uchar y;
  uchar z;
  uchar w;
} uchar4;

typedef struct char1 { char x; } char1;

typedef struct char2 {
  char x;
  char y;
} char2;

typedef struct char3 {
  char x;
  char y;
  char z;
} char3;

typedef struct char4 {
  char x;
  char y;
  char z;
  char w;
} char4;

typedef struct ushort1 { ushort x; } ushort1;

typedef struct ushort2 {
  ushort x;
  ushort y;
} ushort2;

typedef struct ushort3 {
  ushort x;
  ushort y;
  ushort z;
} ushort3;

typedef struct ushort4 {
  ushort x;
  ushort y;
  ushort z;
  ushort w;
} ushort4;

typedef struct short1 { short x; } short1;

typedef struct short2 {
  short x;
  short y;
} short2;

typedef struct short3 {
  short x;
  short y;
  short z;
} short3;

typedef struct short4 {
  short x;
  short y;
  short z;
  short w;
} short4;

typedef struct uint1 { uint x; } uint1;

typedef struct uint2 {
  uint x;
  uint y;
} uint2;

typedef struct uint3 {
  uint x;
  uint y;
  uint z;
} uint3;

typedef struct uint4 {
  uint x;
  uint y;
  uint z;
  uint w;
} uint4;

typedef struct int1 { int x; } int1;

typedef struct int2 {
  int x;
  int y;
} int2;

typedef struct int3 {
  int x;
  int y;
  int z;
} int3;

typedef struct int4 {
  int x;
  int y;
  int z;
  int w;
} int4;

typedef struct ulong1 { ulong x; } ulong1;

typedef struct ulong2 {
  ulong x;
  ulong y;
} ulong2;

typedef struct ulong3 {
  ulong x;
  ulong y;
  ulong z;
} ulong3;

typedef struct ulong4 {
  ulong x;
  ulong y;
  ulong z;
  ulong w;
} ulong4;

typedef struct long1 { long x; } long1;

typedef struct long2 {
  long x;
  long y;
} long2;

typedef struct long3 {
  long x;
  long y;
  long z;
} long3;

typedef struct long4 {
  long x;
  long y;
  long z;
  long w;
} long4;

typedef struct ulonglong1 { ulonglong x; } ulonglong1;

typedef struct ulonglong2 {
  ulonglong x;
  ulonglong y;
} ulonglong2;

typedef struct ulonglong3 {
  ulonglong x;
  ulonglong y;
  ulonglong z;
} ulonglong3;

typedef struct ulonglong4 {
  ulonglong x;
  ulonglong y;
  ulonglong z;
  ulonglong w;
} ulonglong4;

typedef struct longlong1 { longlong x; } longlong1;

typedef struct longlong2 {
  longlong x;
  longlong y;
} longlong2;

typedef struct float1 { float x; } float1;

typedef struct float2 {
  float x;
  float y;
} float2;

typedef struct float3 {
  float x;
  float y;
  float z;
} float3;

typedef struct float4 {
  float x;
  float y;
  float z;
  float w;
} float4;

typedef struct double1 { double x; } double1;

typedef struct double2 {
  double x;
  double y;
} double2;

typedef uint3 dim3;

extern dim3 gridDim;
extern uint3 blockIdx;
extern dim3 blockDim;
extern uint3 threadIdx;
extern int warpsize;

#endif

/*
extern void *cuda_device_buffer;
extern size_t cuda_device_buffer_size;
template <typename T>
T *get_cuda_device_buffer(size_t num_of_elements);

template <typename T>
T *get_cuda_device_buffer(size_t num_of_elements) {
  size_t size = num_of_elements * sizeof(T);
  if (size > cuda_device_buffer_size) {
    if (cuda_device_buffer_size != 0) {
      cudaFree(cuda_device_buffer);
    }
    if (cudaMalloc(&cuda_device_buffer, size) != cudaSuccess)
      abort();
    cuda_device_buffer_size = size;
  }
  return (T *)cuda_device_buffer;
}

const int warp_size = 32;

// Based on
//
https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static double atomicMax(double *address, double val) {
  long long *address_as_ull = (long long *)address;
  long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(
        (unsigned long long *)address_as_ull, (unsigned long long)assumed,
        __double_as_longlong(max(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <typename T>
__inline__ __device__ T reduce_sum_warp(T val) {
  // Reduce the sum of wrap_size threads within a single wrap
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T reduce_max_warp(T val) {
  // Reduce the sum of wrap_size threads within a single wrap
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down(val, offset));
  }
  return val;
}

template <int block_size, typename T>
__inline__ __device__ T reduce_sum_block(T val) {
  __shared__ T shared[block_size / warp_size];
  unsigned int tid = threadIdx.x;
  unsigned int warp_id = tid / warp_size;
  unsigned int lane_id = tid - warp_id * warp_size;
  // Sums in each warp
  val = reduce_sum_warp(val);

  if (lane_id == 0) {
    shared[tid / warp_size] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = shared[lane_id];
    return reduce_sum_warp(val);
  } else {
    return 0;
  }
}

template <int block_size, typename T>
__inline__ __device__ T reduce_max_block(T val) {
  __shared__ T shared[block_size / warp_size];
  unsigned int tid = threadIdx.x;
  unsigned int warp_id = tid / warp_size;
  unsigned int lane_id = tid - warp_id * warp_size;
  // Sums in each warp
  val = reduce_max_warp(val);

  if (lane_id == 0) {
    shared[tid / warp_size] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = shared[lane_id];
    return reduce_max_warp(val);
  } else {
    return 0;
  }
}

template <int block_size, typename T>
__global__ void reduce_sum(const T *__restrict__ g_idata,
                           T *__restrict__ g_odata,
                           unsigned long n) {
  T val = 0;
  // First reduce all numbers into a block
  for (unsigned long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    val += g_idata[i];
  }
  // Then reduce the block
  val = reduce_sum_block<block_size, T>(val);

  // Finally atomic add block results to output
  if (threadIdx.x == 0)
    atomicAdd(g_odata, val);
}

template <int block_size, typename T>
__global__ void reduce_max(const T *__restrict__ g_idata,
                           T *__restrict__ g_odata,
                           unsigned long n) {
  T val = 0;
  // First reduce all numbers into a block
  for (unsigned long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    val = max(val, g_idata[i]);
  }
  // Then reduce the block
  val = reduce_max_block<block_size, T>(val);

  // Finally atomic add block results to output
  if (threadIdx.x == 0)
    atomicMax(g_odata, val);
}
*/
