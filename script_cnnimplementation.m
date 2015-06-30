% Script File: Convolutional Neural Network (CNN) Implementation.
% File to implement the a CNN without backpropagation. 
% Uses MNIST digit database.
%
% (on neuralNetInit1.mat file).
%

% Test convolution on 500
clear all
close all
clc

load neuralNetInit1
warning('off', 'all');

% Initial parameters.
imageDim = [28 28];
numClasses = 10; % because there are 10 digits. 
numImages = 500;
numFilters = 10;
filtDim = 9;
poolSize = 2;

hiddenSize = (poolSize^2)*numFilters;

whichPool = 'var';

% initialize parameters.
theta = rand(filtDim*filtDim*numFilters + numClasses*hiddenSize + ...
    numFilters + numClasses,1);

% initialize images and grounf truth.
images = zeros(imageDim(1), imageDim(2), numImages);
idx = randi(42000, numImages,1);
thisY = Y(:,idx); % this experiment's ground truth.
thisy = y(idx);

for i=1:numImages
    images(:,:,i) = reshape(A(idx(i),:),28,28)';
end

tic 
J = softmaxRegression(theta, images, thisy, filtDim, numFilters, ...
                      numClasses, poolSize, whichPool,false);
t=toc;
fprintf('\nOne iteration of cost function: %f seconds.', t);

tic
options = optimoptions('fminunc','GradObj','on');
[thStar, Jstar, exitflag, output, gradJstar] = fminunc(...
    @(th) softmaxRegression(th, images, thisy, filtDim, numFilters, ...
                      numClasses, poolSize, whichPool,false), ...
                      theta, options);
t2 = toc;
fprintf('\nOptimization without backpropagation: %f seconds.', t2);

