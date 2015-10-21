% script file: 
% Help me pass one 28X28 patch through the neural network with two conv
% layers, two max pooling stages and two fully connected layers. 
% 

clc;

filterDim = 5;
numFilters = 12;
poolsize = 2;

W1 = rand(filterDim, filterDim, numFilters);
b1 = rand(numFilters, 1);
W2 = rand(filterDim, filterDim, 1);
b2 = rand;

A1 = rand(64,192);
A2 = rand(2,64);

X = imread('peppers.png');
X = double(X);
X = rgb2gray(X);

x1 = double(X(110:137,65:92))/max(double(X(:)));
y1 = 1;
x2 = double(X(65:92,110:137))/max(double(X(:)));
y2 = 0;

[Fconv1] = cnnConvolve(filterDim,numFilters,x1,W1,b1);
[Pool1] = cnnPool(poolsize,Fconv1,'mean');
[Fconv2] = cnnConvolve(filterDim, 1,Pool1,W2,b2);
[Pool2] = cnnPool(poolsize,Fconv2,'mean');

netIn = Pool2(:);
a1 = sigmoid(A1*netIn);
a2 = sigmoid(A2*a1);

%% After having this with a dummy patch
dnds = '/Users/jsolisl/Documents/propio/PhD/SW/matlab/CNN/';
imname = 'N2DHGOWT1_0.tif';
bimname = 'GT_N2DHGOWT1_0.tif';

[X, xatt] = readParseInput(strcat(dnds,imname));
[Xb, batt] = readParseInput(strcat(dnds, bimname));

[cells] = getTrainingImages(X,Xb);
Cell = cells(:,:,1);

