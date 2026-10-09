// Toolchain smoke test: device identity plus a checked vector kernel.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) y[i] = a*x[i] + y[i];
}

int main(){
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s, sm_%d%d, %d SMs, %.1f GB\n",
    p.name, p.major, p.minor, p.multiProcessorCount,
    p.totalGlobalMem/1073741824.0);
  const int N = 1 << 20;
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  for(int i = 0; i < N; i++){ x[i] = 1.0f; y[i] = 2.0f; }
  saxpy<<<(N+255)/256, 256>>>(N, 3.0f, x, y);
  cudaDeviceSynchronize();
  double err = 0;
  for(int i = 0; i < N; i++) err += fabs(y[i] - 5.0f);
  printf("saxpy over %d elements, total error %.1f -> %s\n",
    N, err, err == 0 ? "PASS" : "FAIL");
  return err == 0 ? 0 : 1;
}
