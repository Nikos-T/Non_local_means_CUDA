xTPB=16;    %ThreadsPerBlock
yTPB=16;
iterations=5;
% image normalizerC
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

gtx480 = gpuDevice(1);
patchSize = [3 3];
patchSigma = single(0.65);
filtSigmaSquared = single(0.004);
% noise
noiseParams = {'gaussian', 0, 0.001};

load('../data/256.mat');
load('../data/128.mat');
load('../data/64.mat' );

%Preprocess
ori256 = normImg( ori256 );
ori128 = normImg( ori128 );
ori64  = normImg( ori64  );

%add noise
noise256 = imnoise( ori256, noiseParams{:} );
noise128 = imnoise( ori128, noiseParams{:} );
noise64  = imnoise( ori64,  noiseParams{:} );

%%
%start with 256
m=256;
n=256;

%transfer to gpu and pad
noise256 = padarray(gpuArray(noise256), floor(patchSize./2), 'symmetric');

%create matrixes for partial sums and result
w=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
Z=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
filtered256=zeros([m-1, n-1]+patchSize, 'single', 'gpuArray');

%create CUDA objects
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums3PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z12my_reduce256PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB, yTPB, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];

%start test
time256=0;
for i=1:iterations
    tic
    [w, Z] = feval(step1, noise256, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered256=feval(step2, w, Z, filtered256, m, n);
    wait(gtx480);
    time256 =time256+ toc/iterations;
end
%gather results to cpu
noise256 = gather(noise256);
filtered256 = gather(filtered256(1:256,1:254));
%postprocess
filtered256 = normImg(filtered256);

%%
%next is 128
m=128;
n=128;

%transfer to gpu and pad
noise128 = padarray(gpuArray(noise128), floor(patchSize./2), 'symmetric');

%create matrixes for partial sums
w=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
Z=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
filtered128=zeros([m-1, n-1]+patchSize, 'single', 'gpuArray');

%create CUDA objects
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums3PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z11my_reduce64PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB/2, yTPB/2, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];

%start test
time128=0;
for i=1:iterations
    tic
    [w, Z] = feval(step1, noise128, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered128=feval(step2, w, Z, filtered128, m, n);
    wait(gtx480);
    time128 = time128+toc/iterations;
end
%gather results to cpu
noise128 = gather(noise128);
filtered128 = gather(filtered128(1:128, 1:126));
%postprocess
filtered128 = normImg(filtered128);

%%
%finish with 64
m=64;
n=64;
%transfer to gpu and pad
noise64  = padarray(gpuArray(noise64 ), floor(patchSize./2), 'symmetric');

%create matrixes for partial sums
w=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
Z=zeros([m*n, m*n/(xTPB*yTPB)], 'single', 'gpuArray');
filtered64=zeros([m-1, n-1]+patchSize, 'single', 'gpuArray');

%create CUDA objects
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums3PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z11my_reduce16PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB/4, yTPB/4, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];

%start test
time64=0;
for i=1:iterations
    tic
    [w, Z] = feval(step1, noise64, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered64=feval(step2, w, Z, filtered64, m, n);
    wait(gtx480);
    time64 = time64+toc/iterations;
end

%gather results to cpu
noise64 = gather(noise64);
filtered64 = gather(filtered64(1:64,1:62));
%postprocess
filtered64 = normImg(filtered64);
save('../data/data3.mat', 'noise64', 'noise128', 'noise256', 'filtered64', 'filtered128', 'filtered256', 'time64', 'time128', 'time256');
