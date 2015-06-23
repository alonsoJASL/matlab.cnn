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
W = rand(filtDim, filtDim, numFilters);
b = rand(numFilters,1);

idx = randi(42000, numImages,1);

for i=1:numImages
    images(:,:,i) = reshape(A(idx(i),:),28,28)';
end
% now we normalize the filters.
for j=1:numFilters
    W(:,:,j) = W(:,:,j)./sum(sum(W(:,:,j)));
end

%
tic
Fconv = cnnConvolve(filtDim, numFilters, images, W,b);
Fpool = cnnPool(poolSize,Fconv, whichPool);
toc

