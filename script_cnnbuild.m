% Script File: Convolutional Neural Network (CNN) Build.
% File to implement the various steps of a CNN and prove 

% Test convolution on 5000
clear all
close all
clc

load neuralNetInit1

numImages = 50;
numFilters = 7;
filtDim = 8;

poolSize = 10;
whichPool = 'max';

images = zeros(28,28,numImages);
Wcon = rand(filtDim, filtDim, numFilters);
bcon = rand(numFilters,1);

idx = randi(42000, numImages,1);

for i=1:numImages
    images(:,:,i) = reshape(A(idx(i),:),28,28)';
end
% now we normalize the filters.
for j=1:numFilters
    Wcon(:,:,j) = Wcon(:,:,j)./sum(sum(Wcon(:,:,j)));
end

%
tic
Fconv = cnnConvolve(filtDim, numFilters, images, Wcon,bcon);
Fpool = cnnPool(poolSize,Fconv, whichPool);

% Now the neural network layers!
% parameters
epsilon = 0.5; % YOLO
lambda = 1; % YOLO as well!

poolSize2 = size(Fpool);
Funroll = zeros(poolSize2(1)*poolSize2(2)*poolSize2(3), poolSize2(4));

for i=1:size(Funroll,2)
    Funroll(:,i) = reshape(Fpool(:,:,:,i),size(Funroll,1), 1);
end

[M] = size(Funroll,1); % 
thisY = Y(:,idx); % this experiment's ground truth.
thisy = y(idx);

W1 = epsilon.*(2*rand(M,M) - 1);
b1 = epsilon.*(2*rand(M,1) - 1);

W2 = epsilon.*(2*rand(10,M) - 1);
b2 = epsilon.*(2*rand(10,1) - 1);

A1 = Funroll;
A2 = cnnSigmoid(A1,W1,kron(b1,ones(1,numImages)));
A3 = cnnSigmoid(A2, W2, kron(b2, ones(1,numImages)));

H0 = zeros(size(A3));
for i=1:numImages
    H0(A3(:,i)==max(A3(:,i)),i) = 1;
end

vectJ = (thisY.*log(A3) + (1-thisY).*log(1-A3));
J = -sum(vectJ(:))/numImages + ...
    lambda*(sum(W1(:)) + sum(W2(:)))/(2*numImages);

delta3 = H0 - thisY;
gZ2 = A2.*(1-A2);
delta2 = (W2'*delta3).*gZ2;

DELTA1 = delta2*A1';
DELTA2 = delta3*A2';

%
D1 = DELTA1' + lambda.*W1;
D2 = DELTA2 + lambda.*W2;

toc