xTPB=16;    %ThreadsPerBlock
yTPB=16;

% image normalizerC
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

gtx480 = gpuDevice(1);
patchSize = [5 5];
patchSigma = single(5/3);
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
step1=parallel.gpu.CUDAKernel('../cuda/nlm_step1.ptx', '../cuda/nlm_step1.cu', '_Z18calc_partial_sums5PKfPfS1_ffii');
step1.ThreadBlockSize=[xTPB, yTPB, 1];
step1.GridSize=[m/xTPB, n/yTPB, min(m*n, 65535)];

step2=parallel.gpu.CUDAKernel('../cuda/nlm_step2.ptx', '../cuda/nlm_step2.cu', '_Z11my_reduce16PKfS0_Pfii');
step2.ThreadBlockSize=[xTPB/4, yTPB/4, 1];
step2.GridSize=[1, 1, min(m*n, 65535)];

%start
i=1;
for filtSigmaSquared=single(0.004:0.0005:0.006)
    [w, Z] = feval(step1, noise64, w, Z, patchSigma, filtSigmaSquared, m, n);
    filtered64=feval(step2, w, Z, filtered64, m, n);


%gather results to cpu
noise64 = gather(noise64);
filteredt64 = gather(filtered64);
%postprocess
filteredt64 = normImg(filteredt64(1:64, 1:60));

[peaksnr64_5(i), snr64_5(i)] = psnr(filteredt64, ori64(1:64, 1:60));
i=i+1;
end
%save('../data/data5.mat', 'noise64', 'noise128', 'noise256', 'filtered64', 'filtered128', 'filtered256', 'time64', 'time128', 'time256');

