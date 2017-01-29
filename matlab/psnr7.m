xTPB=16;    %ThreadsPerBlock
yTPB=16;

% image normalizerC
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

gtx480 = gpuDevice(1);
patchSize = [7 7];
patchSigma = single(7/3);


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
i=1;
for filtSigmaSquared=single(0.00005:0.00005:0.0005)
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
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums7PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z12my_reduce256PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB, yTPB, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];


    [w, Z] = feval(step1, noise256, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered256=feval(step2, w, Z, filtered256, m, n);

%gather results to cpu
noise256 = gather(noise256);
filtered256 = gather(filtered256);
%postprocess
filtered256 = normImg(filtered256);
[peaksnr256(i),snr256(i)] = psnr(filtered256(1:256,1:250), ori256(1:256, 1:250));

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
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums7PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z11my_reduce64PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB/2, yTPB/2, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];



    [w, Z] = feval(step1, noise128, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered128=feval(step2, w, Z, filtered128, m, n);
    wait(gtx480);

%gather results to cpu
noise128 = gather(noise128);
filtered128 = gather(filtered128);
%postprocess
filtered128 = normImg(filtered128);
[peaksnr128(i),snr128(i)] = psnr(filtered128(1:128, 1:122), ori128(1:128,1:122));

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
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums7PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z11my_reduce16PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB/4, yTPB/4, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];


    [w, Z] = feval(step1, noise64, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered64=feval(step2, w, Z, filtered64, m, n);

%gather results to cpu
noise64 = gather(noise64);
filtered64 = gather(filtered64);
%postprocess
filtered64 = normImg(filtered64);
[peaksnr64(i),snr64(i)] = psnr(filtered64(1:64,1:58), ori64(1:64,1:58));
i=i+1;
end
save('../data/psnr7.mat', 'filteredm64', 'filteredm128', 'filteredm256','peaksnr64', 'peaksnr128', 'peaksnr256', 'snr256', 'snr128', 'snr64');

