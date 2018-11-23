// Upgraded to global-illumination by Yuanming Hu

#include <helper_cuda.h>
#include <helper_math.h>
#include "util.h"
#include "volume_renderer.h"
#include <curand.h>
#include <curand_kernel.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;

typedef unsigned char VolumeType;
// typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;  // 3D texture

#include <unistd.h>
#include <stdio.h>

__device__ curandState_t states[1024 * 1024 * 40];
__device__ bool mirroring[3];

__global__ void init_random_numbers(unsigned int seed) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  int idx = threadId;
  curand_init(idx + (seed * 1000000007), 0, 0, &states[idx]);
}

// It seems that it will be faster, if we grab states to local memory?
__device__ float randf() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  int idx = threadId;
  return curand_uniform(&states[idx]);
}

typedef struct { float4 m[3]; } float3x4;
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
  float3 o;  // origin
  float3 d;  // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(Ray r,
                            float3 boxmin,
                            float3 boxmax,
                            float *tnear,
                            float *tfar) {
  // compute intersection of ray with all six bbox planes
  float3 invR = make_float3(1.0f) / r.d;
  float3 tbot = invR * (boxmin - r.o);
  float3 ttop = invR * (boxmax - r.o);

  // re-order intersections to find smallest and largest on each axis
  float3 tmin = fminf(ttop, tbot);
  float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v) {
  float3 r;
  r.x = dot(v, make_float3(M.m[0]));
  r.y = dot(v, make_float3(M.m[1]));
  r.z = dot(v, make_float3(M.m[2]));
  return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v) {
  float4 r;
  r.x = dot(v, M.m[0]);
  r.y = dot(v, M.m[1]);
  r.z = dot(v, M.m[2]);
  r.w = 1.0f;
  return r;
}

__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
         (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__constant__ float slice_position;
// Actually, half box size
__constant__ float3 raw_box_size;
__constant__ float3 raw_box_res;
__device__ float3 box_scale;

__inline__ __device__ float3 get_box_size() {
  return raw_box_size * box_scale;
}

__inline__ __device__ float3 get_box_res() {
  return raw_box_res * box_scale;
}

// -1 ~ 1
__device__ bool inside_unit_cube(float3 orig) {
  return -1.0f < orig.x && orig.x < 1 && -1.0f < orig.y && orig.y < 1.0f &&
         -1.0f < orig.z && orig.z < 1.0f;
}

__device__ float fract(float x) {
  return x - floorf(x);
}

const int maxSteps = 1025;

const float tstep = 0.01f;

inline __device__ float3 wrap_coord(float3 pos) {
  pos = (pos * 0.5 + 0.5) * box_scale;
  if (pos.x > 1)
    pos.x = 2 - pos.x;
  if (pos.y > 1)
    pos.y = 2 - pos.y;
  if (pos.z > 1)
    pos.z = 2 - pos.z;
  return pos * 2 - 1;
}

inline __device__ float sample_tex(float3 pos) {
  pos = pos / get_box_size();
  if (!inside_unit_cube(pos) || pos.z > 2 * slice_position - 1) {
    return 0;
  }
  pos = wrap_coord(pos);
  return tex3D(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
               pos.z * 0.5f + 0.5f);
}

// pos should be in 0~box_res
inline __device__ float sample_tex_int(float3 pos) {
  pos = (pos + make_float3(0.5f)) / get_box_res() * 2 - 1;
  // -1 ~ 1
  if (!inside_unit_cube(pos) || pos.z > 2 * slice_position - 1) {
    return 0;
  }
  pos = wrap_coord(pos);
  return tex3D(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
               pos.z * 0.5f + 0.5f);
}

inline __device__ float3 sample_sphere(float u, float v) {
  float x = u * 2 - 1;
  float phi = v * 2 * 3.14159265358979f;
  float yz = sqrt(1 - x * x);
  return make_float3(x, yz * cos(phi), yz * sin(phi));
}

#define DDA

#ifndef DDA
__device__ void get_next_hit(Ray eyeRay,
                             bool &hit,
                             float3 &hit_pos,
                             float density,
                             bool need_position = true) {
  float tnear, tfar;
  hit = (bool)intersectBox(eyeRay, -box_size, box_size, &tnear, &tfar);
  if (!hit) {
    hit = false;
    return;
  }

  if (tnear < 0.0f)
    tnear = 0.0f;  // clamp to near plane

  float3 pos = eyeRay.o + eyeRay.d * (tnear + 1e-4f);
  float3 step = eyeRay.d * tstep;

  for (int i = 0; i < maxSteps; i++) {
    if (sample_tex(pos) > density) {
      float3 back_step = step;
      for (int k = 0; k < 30 * int(need_position); k++) {
        back_step *= 0.5;
        if (sample_tex(pos - back_step) > density) {
          // Go back until we touch the air
          pos -= back_step;
        }
      }
      hit_pos = pos;
      hit = true;
      return;
    }
    pos += step;
  }
  hit = false;
}
#endif

__device__ float3 floor(float3 v) {
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

__device__ float sign(float x)

{
  return x > 0 ? 1 : (x < 0 ? -1 : 0);
}

__device__ float3 sign(float3 v) {
  return make_float3(sign(v.x), sign(v.y), sign(v.z));
}

#ifdef DDA
// https://www.shadertoy.com/view/4dfGzs
__device__ float get_next_hit(Ray eyeRay,
                             bool &hit,
                             float3 &hit_pos,
                             float density,
                             bool need_position = true) {
  eyeRay.d = eyeRay.d / sqrt(eyeRay.d.x * eyeRay.d.x + eyeRay.d.y * eyeRay.d.y +
                             eyeRay.d.z * eyeRay.d.z);
  float tnear, tfar;
  float3 box_size = get_box_size();
  float3 box_res = get_box_res();
  hit = (bool)intersectBox(eyeRay, -box_size, box_size, &tnear, &tfar);
  if (!hit) {
    hit = false;
    return;
  }

  if (tnear < 0.0f)
    tnear = 0.0f;  // clamp to near plane

  float3 pos = eyeRay.o + eyeRay.d * (tnear + 1e-4f);
  float3 step = eyeRay.d * tstep;

  float3 ri = 1.0f / eyeRay.d;
  float3 rs = sign(eyeRay.d);
  // 0 to raw_box_res - 1
  float3 o = (pos + box_size) * 0.5f * (box_res.x / box_size.x);
  float3 ipos = floor(o);
  float3 dis = (ipos - o + 0.5f + rs * 0.5f) * ri;

  float last_sample = 0;
  for (int i = 0; i < maxSteps; i++) {
    last_sample = sample_tex_int(ipos);
    if (last_sample > density) {
      // intersect the cube
      float3 mini = (ipos - o + 0.5 - rs * 0.5f) * ri;
      float t = max(mini.x, max(mini.y, mini.z)) * (box_size.x / box_res.x) * 2;
      hit_pos = pos + t * eyeRay.d;
      hit = true;
      return;
    }

    float3 mm;
    if (dis.x <= dis.y && dis.x < dis.z) {
      mm = make_float3(1, 0, 0);
    } else if (dis.y <= dis.x && dis.y <= dis.z) {
      mm = make_float3(0, 1, 0);
    } else {
      mm = make_float3(0, 0, 1);
    }
    dis += mm * rs * ri;
    ipos += mm * rs;
  }
  hit = false;
  return last_sample;
}
#endif

__device__ float3 trace(Ray eyeRay,
                        int limit,
                        int sample_count,
                        float density) {
  float3 sum = make_float3(0.86, 0.85, 0.9);

  bool hit;
  float3 pos;

  float c = get_next_hit(eyeRay, hit, pos, density, true);
  if (hit) {
    // sum = pos * 0.5 + 0.5;
    // return sum;
    sum *= 0;
    int samples = 3;
    Ray occ;
    occ.o = pos;
    for (int j = 0; j < samples; j++) {
      float3 direction = sample_sphere(randf(), randf());
      occ.d = direction;
      bool next_hit;
      float3 _;
      get_next_hit(occ, next_hit, _, density, true);
      float3 coeff1 = clamp(
          make_float3(dot(direction, make_float3(0.6, 0.75, 0.15))) * 0.5f +
              0.5f,
          0.0f, 1.0f);
      float3 light = (coeff1 * make_float3(0.9, 0.7, 0.3) +
                      (1 - coeff1) * make_float3(0.4, 0.5, 0.9));

      if (!next_hit) {
        sum += (2.0f / samples) * light;
      }
    }
#ifdef WIREFRAME
    if (c > 0.99) {
      sum *= 0.6;
      sum += make_float3(0.3, 0., 0);
    }
#endif
  }
  return sum;
}

__device__ inline float4 sqrt(float4 f) {
  return make_float4(sqrt(f.x), sqrt(f.y), sqrt(f.z), sqrt(f.w));
}

__device__ inline float3 max(float3 a, float3 b) {
  return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__global__ void render_surface(uint *d_output,
                               uint imageW,
                               uint imageH,
                               float density,
                               float *buffer,
                               int counter) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= imageW) || (y >= imageH))
    return;

  float u = ((x + randf()) / (float)imageW) * 2.0f - 1.0f;
  float v = ((y + randf()) / (float)imageW) * 2.0f - 1.0f;

  // calculate eye ray in world space
  Ray eyeRay;
  eyeRay.o =
      make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
  eyeRay.d = normalize(make_float3(u, v, -2.0f));
  eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
  // write output color
  float3 pixel = trace(eyeRay, 3, counter, density) * 1;
  buffer[(y * imageW + x) * 3 + 0] += pixel.x;
  buffer[(y * imageW + x) * 3 + 1] += pixel.y;
  buffer[(y * imageW + x) * 3 + 2] += pixel.z;

  float darkness = 1.0f - 0.15f * (u * u + v * v);

  float scale = darkness;

  float3 color = make_float3(buffer[(y * imageW + x) * 3 + 0],
                             buffer[(y * imageW + x) * 3 + 1],
                             buffer[(y * imageW + x) * 3 + 2]) /
                 make_float3(counter);
  // http://filmicworlds.com/blog/filmic-tonemapping-operators/
  /*
  float3 X = max(make_float3(0.0f), color - 0.004f);
  float3 retColor = (X * (6.2f * X + .5f)) / (X * (6.2f * X + 1.7f) + 0.06f);
  */
  float3 retColor = color;
  d_output[y * imageW + x] = rgbaFloatToInt(scale * make_float4(retColor, 1));
}

__global__ void render_volume(uint *d_output,
                              uint imageW,
                              uint imageH,
                              float density,
                              float *buffer,
                              int counter,
                              float thres) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= imageW) || (y >= imageH))
    return;

  float u = ((x + randf()) / (float)imageW) * 2.0f - 1.0f;
  float v = ((y + randf()) / (float)imageW) * 2.0f - 1.0f;

  // calculate eye ray in world space
  Ray eyeRay;
  eyeRay.o =
      make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
  eyeRay.d = normalize(make_float3(u, v, -2.0f));
  eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

  float sum = 0;

  float tnear, tfar;
  float3 box_size = get_box_size();
  bool hit = (bool)intersectBox(eyeRay, -box_size, box_size, &tnear, &tfar);
  if (hit) {
    if (tnear < 0.0f)
      tnear = 0.0f;  // clamp to near plane

    float3 pos = eyeRay.o + eyeRay.d * (tnear + 1e-4f);
    float3 step = eyeRay.d * tstep;
    for (int i = 0; i < maxSteps; i++) {
      sum += sample_tex(pos + randf() * step);
      pos += step;
    }
  }
  sum *= 1e-2;
  sum = max(sum - thres * 1, 0.0f);
  //density = density / (0.1 + density);
  float3 pixel = 1 - make_float3(1 - exp(-density * sum));

  buffer[(y * imageW + x) * 3 + 0] += pixel.x;
  buffer[(y * imageW + x) * 3 + 1] += pixel.y;
  buffer[(y * imageW + x) * 3 + 2] += pixel.z;

  float darkness = 1.0f - 0.15f * (u * u + v * v);

  float scale = darkness;

  float3 color = make_float3(buffer[(y * imageW + x) * 3 + 0],
                             buffer[(y * imageW + x) * 3 + 1],
                             buffer[(y * imageW + x) * 3 + 2]) /
                 make_float3(counter);
  float3 retColor = color;
  d_output[y * imageW + x] = rgbaFloatToInt(scale * make_float4(retColor, 1));
}

void setTextureFilterMode(bool bLinearFilter) {
  tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

void update_volume(void *h_volume, cudaExtent volumeSize) {
  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType),
                          volumeSize.width, volumeSize.height);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));
}

void initCuda(void *h_volume, cudaExtent volumeSize) {
  // create 3D array
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
  checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

  update_volume(h_volume, volumeSize);

  // set texture parameters
  tex.normalized = true;  // access with normalized texture coordinates
  tex.filterMode = cudaFilterModePoint;       // no interpolation initially
  tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  tex.addressMode[1] = cudaAddressModeClamp;

  // bind array to 3D texture
  checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

void freeCudaBuffers() {
  checkCudaErrors(cudaFreeArray(d_volumeArray));
}

float *render_buffer = NULL;
float last_width = 0, last_height = 0;
int sample_counter = 0;

void reset_render_buffer(int width, int height) {
  sample_counter = 0;
  bool different_size = last_width != width || last_height != height;
  if (render_buffer && different_size) {
    checkCudaErrors(cudaFree(render_buffer));
    render_buffer = NULL;
  }
  last_width = width;
  last_height = height;
  uint buffer_size = sizeof(float) * width * height * 3;
  if (!render_buffer)
    checkCudaErrors(cudaMalloc(&render_buffer, buffer_size));
  checkCudaErrors(cudaMemset(render_buffer, 0, buffer_size));
}

void set_box_size(float *size) {
  checkCudaErrors(cudaMemcpyToSymbol(raw_box_size, size, sizeof(raw_box_size)));
}

__global__ void update_box_scale() {
  box_scale.x = 1 + (int)mirroring[0];
  box_scale.y = 1 + (int)mirroring[1];
  box_scale.z = 1 + (int)mirroring[2];
}

void set_mirroring(bool *new_mirroring) {
  checkCudaErrors(
      cudaMemcpyToSymbol(mirroring, new_mirroring, sizeof(mirroring)));
  update_box_scale<<<1, 1>>>();
}

void set_box_res(int *res) {
  float fres[3];
  fres[0] = res[0];
  fres[1] = res[1];
  fres[2] = res[2];
  checkCudaErrors(cudaMemcpyToSymbol(raw_box_res, fres, sizeof(raw_box_res)));
}

void render_kernel(dim3 gridSize,
                   dim3 blockSize,
                   uint *d_output,
                   uint imageW,
                   uint imageH,
                   float density,
                   float slice,
                   float volume_rendering) {
  /*
  curandState_t *states_p;
  checkCudaErrors(cudaMalloc((void **)&states_p, gridSize.x * gridSize.y *
                                                     blockSize.x * blockSize.y *
                                                     sizeof(curandState_t)));
  cudaMemcpyToSymbol("states", &states_p, sizeof(curandState_t *),
  cudaMemcpyHostToDevice);
  // cudaDeviceSynchronize();
  */
  sample_counter += 1;
  init_random_numbers<<<gridSize, blockSize>>>(sample_counter);
  checkCudaErrors(cudaMemcpyToSymbol(slice_position, &slice, sizeof(slice)));
  if (!volume_rendering) {
    render_surface<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                            render_buffer, sample_counter);
  } else {
    render_volume<<<gridSize, blockSize>>>(d_output, imageW, imageH,
                                           volume_rendering, render_buffer,
                                           sample_counter, density);
  }
  // cudaFree(states_p);
}

void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) {
  checkCudaErrors(
      cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}
