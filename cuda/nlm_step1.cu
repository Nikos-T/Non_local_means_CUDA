/* blockDim=[16, 16, 1]
 * 
 * i refers to x coordinate of the j or y pixel
 * j refers to y coordinate of the j or y pixel
 * k refers to x coordinate of the i or x pixel
 * l refers to y coordinate of the i or x pixel
 * */

__global__ void calc_partial_sums7(float const * const A, float *partialSum16x16, float *ZpartialSum16x16, float patchSigma, float filtSigmaSquared, int m, int n) {
  
  __shared__ float Z[256], my_w[256];
  float differences_of_areaG[49];
  
  int i=blockDim.x*blockIdx.x+threadIdx.x+3;
  int j=blockDim.y*blockIdx.y+threadIdx.y+3;
  int k=(blockIdx.z)%(m+6)+3;
  int l=(blockIdx.z)/(m+6)+3;
  int tid=threadIdx.y*16+threadIdx.x;
  
  for (int u=-3; u<=3; u++) {
    for (int v=-3; v<=3; v++) {
      differences_of_areaG[7*(u+3)+v+3]=(A[(l+u)*(m+6)+k+v]-A[(j+u)*(m+6)+i+v])*expf(-(u*u+v*v)/(2*patchSigma)); // The first A is the same for the whole block.
    }
  }
  Z[tid]=expf(-powf(normf(49, differences_of_areaG),2)/(2*filtSigmaSquared));
  my_w[tid]=Z[tid]*A[i+j*(m+6)];
  // Reduction algorithm
  __syncthreads();
  if (tid<128) {Z[tid]+=Z[tid+128]; my_w[tid]+=my_w[tid+128]; } __syncthreads();
  if (tid< 64) {Z[tid]+=Z[tid+ 64]; my_w[tid]+=my_w[tid+ 64]; } __syncthreads();
  if (tid< 32) {Z[tid]+=Z[tid+ 32]; my_w[tid]+=my_w[tid+ 32]; } __syncthreads();
  if (tid< 16) {Z[tid]+=Z[tid+ 16]; my_w[tid]+=my_w[tid+ 16]; } __syncthreads();
  if (tid<  8) {Z[tid]+=Z[tid+  8]; my_w[tid]+=my_w[tid+  8]; } __syncthreads();
  if (tid<  4) {Z[tid]+=Z[tid+  4]; my_w[tid]+=my_w[tid+  4]; } __syncthreads();
  if (tid<  2) {Z[tid]+=Z[tid+  2]; my_w[tid]+=my_w[tid+  2]; } __syncthreads();
  if (tid<  1) {
    // Pass the value to global variable
    ZpartialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=Z[0]+Z[1];
     partialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=my_w[0]+my_w[1];
  }
}

/* blockDim=[16, 16, 1]
 * 
 * i refers to x coordinate of the j or y pixel
 * j refers to y coordinate of the j or y pixel
 * k refers to x coordinate of the i or x pixel
 * l refers to y coordinate of the i or x pixel
 * */

__global__ void calc_partial_sums5(float const * const A, float *partialSum16x16, float *ZpartialSum16x16, float patchSigma, float filtSigmaSquared, int m, int n) {
  
  __shared__ float Z[256], my_w[256];
  float differences_of_areaG[25];
  
  int i=blockDim.x*blockIdx.x+threadIdx.x+2;
  int j=blockDim.y*blockIdx.y+threadIdx.y+2;
  int k=(blockIdx.z)%(m+4)+2;
  int l=(blockIdx.z)/(m+4)+2;
  int tid=threadIdx.y*16+threadIdx.x;
  
  for (int u=-2; u<=2; u++) {
    for (int v=-2; v<=2; v++) {
      differences_of_areaG[7*(u+2)+v+2]=(A[(l+u)*(m+4)+k+v]-A[(j+u)*(m+4)+i+v])*expf(-(u*u+v*v)/(2*patchSigma)); // The first A is the same for the whole block.
    }
  }
  Z[tid]=expf(-powf(normf(25, differences_of_areaG),2)/(2*filtSigmaSquared));
  my_w[tid]=Z[tid]*A[i+j*(m+4)];
  // Reduction algorithm
  __syncthreads();
  if (tid<128) {Z[tid]+=Z[tid+128]; my_w[tid]+=my_w[tid+128]; } __syncthreads();
  if (tid< 64) {Z[tid]+=Z[tid+ 64]; my_w[tid]+=my_w[tid+ 64]; } __syncthreads();
  if (tid< 32) {Z[tid]+=Z[tid+ 32]; my_w[tid]+=my_w[tid+ 32]; } __syncthreads();
  if (tid< 16) {Z[tid]+=Z[tid+ 16]; my_w[tid]+=my_w[tid+ 16]; } __syncthreads();
  if (tid<  8) {Z[tid]+=Z[tid+  8]; my_w[tid]+=my_w[tid+  8]; } __syncthreads();
  if (tid<  4) {Z[tid]+=Z[tid+  4]; my_w[tid]+=my_w[tid+  4]; } __syncthreads();
  if (tid<  2) {Z[tid]+=Z[tid+  2]; my_w[tid]+=my_w[tid+  2]; } __syncthreads();
  if (tid<  1) {
    // Pass the value to global variable
    ZpartialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=Z[0]+Z[1];
     partialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=my_w[0]+my_w[1];
  }
}

__global__ void calc_partial_sums3(float const * const A, float *partialSum16x16, float *ZpartialSum16x16, float patchSigma, float filtSigmaSquared, int m, int n) {
  
  __shared__ float Z[256], my_w[256];
  float differences_of_areaG[9];
  
  int i=blockDim.x*blockIdx.x+threadIdx.x+1;
  int j=blockDim.y*blockIdx.y+threadIdx.y+1;
  int k=(blockIdx.z)%(m+2)+1;
  int l=(blockIdx.z)/(m+2)+1;
  int tid=threadIdx.y*16+threadIdx.x;
  
  for (int u=-1; u<=1; u++) {
    for (int v=-1; v<=1; v++) {
      differences_of_areaG[7*(u+1)+v+1]=(A[(l+u)*(m+2)+k+v]-A[(j+u)*(m+2)+i+v])*expf(-(u*u+v*v)/(2*patchSigma)); // The first A is the same for the whole block.
    }
  }
  Z[tid]=expf(-powf(normf(9, differences_of_areaG),2)/(2*filtSigmaSquared));
  my_w[tid]=Z[tid]*A[i+j*(m+2)];
  // Reduction algorithm
  __syncthreads();
  if (tid<128) {Z[tid]+=Z[tid+128]; my_w[tid]+=my_w[tid+128]; } __syncthreads();
  if (tid< 64) {Z[tid]+=Z[tid+ 64]; my_w[tid]+=my_w[tid+ 64]; } __syncthreads();
  if (tid< 32) {Z[tid]+=Z[tid+ 32]; my_w[tid]+=my_w[tid+ 32]; } __syncthreads();
  if (tid< 16) {Z[tid]+=Z[tid+ 16]; my_w[tid]+=my_w[tid+ 16]; } __syncthreads();
  if (tid<  8) {Z[tid]+=Z[tid+  8]; my_w[tid]+=my_w[tid+  8]; } __syncthreads();
  if (tid<  4) {Z[tid]+=Z[tid+  4]; my_w[tid]+=my_w[tid+  4]; } __syncthreads();
  if (tid<  2) {Z[tid]+=Z[tid+  2]; my_w[tid]+=my_w[tid+  2]; } __syncthreads();
  if (tid<  1) {
    // Pass the value to global variable
    ZpartialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=Z[0]+Z[1];
     partialSum16x16[gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x]=my_w[0]+my_w[1];
  }
}
