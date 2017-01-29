/* blockDim=[16, 16, 1]
 * 
 * 
 * p refers to the new pixel
 * */

__global__ void my_reduce256(float const * const partialSum16x16, float const * const ZpartialSum16x16, float *filtI, int m, int n) {
  
  __shared__ float Z[256], my_w[256];
  int p=/*blockDim.z*blockIdx.z+threadIdx.z*/blockIdx.z;
  int tid = 16*threadIdx.y+threadIdx.x;
  Z[tid]=ZpartialSum16x16[p*256+tid];
  my_w[tid]=partialSum16x16[p*256+tid];
  __syncthreads();
  if (tid<128) {my_w[tid]+=my_w[tid+128]; Z[tid]+=Z[tid+128]; } __syncthreads();
  if (tid< 64) {my_w[tid]+=my_w[tid+ 64]; Z[tid]+=Z[tid+ 64]; } __syncthreads();
  if (tid< 32) {my_w[tid]+=my_w[tid+ 32]; Z[tid]+=Z[tid+ 32]; } __syncthreads();
  if (tid< 16) {my_w[tid]+=my_w[tid+ 16]; Z[tid]+=Z[tid+ 16]; } __syncthreads();
  if (tid<  8) {my_w[tid]+=my_w[tid+  8]; Z[tid]+=Z[tid+  8]; } __syncthreads();
  if (tid<  4) {my_w[tid]+=my_w[tid+  4]; Z[tid]+=Z[tid+  4]; } __syncthreads();
  if (tid<  2) {my_w[tid]+=my_w[tid+  2]; Z[tid]+=Z[tid+  2]; } __syncthreads();
  if (tid<  1) {
    filtI[p]=(my_w[0]+my_w[1])/(Z[0]+Z[1]);
  }
}

__global__ void my_reduce64(float const * const partialSum16x16, float const * const ZpartialSum16x16, float *filtI, int m, int n) {
  
  __shared__ float Z[64], my_w[64];
  int p=/*blockDim.z*blockIdx.z+threadIdx.z*/blockIdx.z;
  int tid = 8*threadIdx.y+threadIdx.x;
  Z[tid]=ZpartialSum16x16[p*64+tid];
  my_w[tid]=partialSum16x16[p*64+tid];
  __syncthreads();
  if (tid< 32) {my_w[tid]+=my_w[tid+ 32]; Z[tid]+=Z[tid+ 32]; } __syncthreads();
  if (tid< 16) {my_w[tid]+=my_w[tid+ 16]; Z[tid]+=Z[tid+ 16]; } __syncthreads();
  if (tid<  8) {my_w[tid]+=my_w[tid+  8]; Z[tid]+=Z[tid+  8]; } __syncthreads();
  if (tid<  4) {my_w[tid]+=my_w[tid+  4]; Z[tid]+=Z[tid+  4]; } __syncthreads();
  if (tid<  2) {my_w[tid]+=my_w[tid+  2]; Z[tid]+=Z[tid+  2]; } __syncthreads();
  if (tid<  1) {
    filtI[p]=(my_w[0]+my_w[1])/(Z[0]+Z[1]);
  }
}

__global__ void my_reduce16(float const * const partialSum16x16, float const * const ZpartialSum16x16, float *filtI, int m, int n) {
  
  __shared__ float Z[16], my_w[16];
  int p=/*blockDim.z*blockIdx.z+threadIdx.z*/blockIdx.z;
  int tid = 4*threadIdx.y+threadIdx.x;
  Z[tid]=ZpartialSum16x16[p*16+tid];
  my_w[tid]=partialSum16x16[p*16+tid];
  __syncthreads();
  if (tid<  8) {my_w[tid]+=my_w[tid+  8]; Z[tid]+=Z[tid+  8]; } __syncthreads();
  if (tid<  4) {my_w[tid]+=my_w[tid+  4]; Z[tid]+=Z[tid+  4]; } __syncthreads();
  if (tid<  2) {my_w[tid]+=my_w[tid+  2]; Z[tid]+=Z[tid+  2]; } __syncthreads();
  if (tid<  1) {
    filtI[p]=(my_w[0]+my_w[1])/(Z[0]+Z[1]);
  }
}
