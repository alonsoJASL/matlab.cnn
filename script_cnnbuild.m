% Script File: Convolutional Neural Network (CNN) Build.
% File to implement the various steps of a CNN. Uses MNIST digit database
% (on neuralNetInit1.mat file).
% 

% Test convolution on 500
clear all
close all
clc

load neuralNetInit1
tic 

imageDim = [28 28];
numClasses = 10; % because there are 10 digits. 
numImages = 500;
numFilters = 10;
filtDim = 9;
poolSize = 2;

whichPool = 'var';

hiddenSize = (poolSize^2)*numFilters;
% parameters for the implementation of the neural network.
epsilon = 0.01; % YOLO
lambda = 1; % YOLO as well!

% Images and groud truth
images = zeros(imageDim(1), imageDim(2), numImages);
idx = randi(42000, numImages,1);
thisY = Y(:,idx); % this experiment's ground truth.
thisy = y(idx);

for i=1:numImages
    images(:,:,i) = reshape(A(idx(i),:),28,28)';
end

Wcon = rand(filtDim, filtDim, numFilters);
bcon = rand(numFilters,1);
Wd = rand(numClasses, hiddenSize);
bd = rand(numClasses,1);

%
Fconv = cnnConvolve(filtDim, numFilters, images, Wcon,bcon);
[convDim1, convDim2,~,~] = size(Fconv);
Fpool = cnnPool(poolSize,Fconv, whichPool);

% Now the neural network layers
% Unrolling the actual input to the neural network. 
Funroll = zeros(hiddenSize, numImages);
%
for i=1:numImages
    Funroll(:,i) = reshape(Fpool(:,:,:,i),hiddenSize, 1);
end

%Forward propagation
A1 = Funroll;
A2 = cnnSigmoid(A1,Wd,bd);

h0 = zeros(numImages,1);
for i=1:numImages
    [~,h0(i)] = max(A2(:,i));
end

% DO THE SOFTMAX REGRESSION
yA2 = zeros(size(thisy));
expA2 = A2;
for i=1:numImages
    yA2(i) = expA2(thisy(i),i);
end
sumA2 = sum(expA2,1)'; 
%
J = - sum(yA2./sumA2);

toc
%% The old way
[M] = size(Funroll,1); % 

W1 = epsilon.*(2*rand(M,M) - 1);
b1 = epsilon.*(2*rand(M,1) - 1);

W2 = epsilon.*(2*rand(10,M) - 1);
b2 = epsilon.*(2*rand(10,1) - 1);

A1 = Funroll;
A2 = cnnSigmoid(A1,W1,b1);
A3 = cnnSigmoid(A2, W2,b2);

H0 = zeros(size(A3));
h0 = zeros(size(thisy));
for i=1:numImages
    H0(A3(:,i)==max(A3(:,i)),i) = 1;
    [~,h0(i)] = max(A3(:,i));
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
