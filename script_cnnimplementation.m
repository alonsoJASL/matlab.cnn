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
numImages = 200;
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
[J, gradJ] = softmaxRegression(theta, images, thisy, filtDim, ...
                        numFilters, numClasses, poolSize, whichPool,false);
t=toc;
timeDisplay(t, 'Cost function'); % function part of the UTILS repo
%%

tic 
% minimisation with our computedd gradient
options = optimoptions('fminunc','GradObj','on', 'Display', 'off',...
    'TolFun', 1e-8, 'TolX', 1e-8);
[thStar, Jstar, exitflag, output, GRADj] = fminunc(@(t) softmaxRegression(...
    t,images,thisy,filtDim,numFilters,numClasses,poolSize,whichPool, false),...
    theta, options);
t2 = toc;
timeDisplay(t2, 'Optimised with fminunc'); % function part of the UTILS repo

%% With our own BFGS algorithm
clc

tic 
[Xbfgs, ~, kfail] = BFGS('softmaxRegression',theta,100,images, thisy, ...
    filtDim, numFilters, numClasses, poolSize, whichPool, false);
%
[Jbfgs, gradJbfgs] =  softmaxRegression(Xbfgs, images, thisy, filtDim, ...
                        numFilters, numClasses, poolSize, whichPool,false);
t3 = toc;
timeDisplay(t3, 'BFGS'); % function part of the UTILS repo

%% using PSO
tic 
[thPSO, Jpso] = genericpso('softmaxRegression', [], theta, 1e-4, 30, ...
    images, thisy, filtDim, numFilters, numClasses, poolSize, whichPool,false)
                    
%% check results
clc
timeDisplay(t2, 'Optimised with fminunc'); % function part of the UTILS repo
timeDisplay(t3, 'BFGS'); % function part of the UTILS repo

valtheta = cnnTestone(theta, images, ...
    filtDim, numFilters, numClasses, poolSize, whichPool);
valStar= cnnTestone(thStar, images, ...
    filtDim, numFilters, numClasses, poolSize, whichPool);
valBFGS = cnnTestone(Xbfgs, images, ...
    filtDim, numFilters, numClasses, poolSize, whichPool);


disp('|Truth|original|fminunc|BFGS|');
disp([thisy valtheta' valStar' valBFGS']);

%%
close all

[Wcstar, Wdstar, bcstar, bdstar] = cnnUnfoldParameters(thStar, filtDim, ...
     numFilters, numClasses, poolSize);
[Wc, Wd, bc, bd] = cnnUnfoldParameters(theta, filtDim, ...
     numFilters, numClasses, poolSize);
[difWc, difWd, difbc, difbd] = cnnUnfoldParameters(abs(thStar-theta), filtDim, ...
     numFilters, numClasses, poolSize);
 
 figure
 for i=1:size(Wcstar,3)
     subplot(2,5,i);
     imagesc(Wcstar(:,:,i));
 end
 figure
 for i=1:size(Wc,3)
     subplot(2,5,i);
     imagesc(Wc(:,:,i));
 end
 
  figure
 for i=1:size(difWc,3)
     subplot(2,5,i);
     imagesc(difWc(:,:,i));
 end
