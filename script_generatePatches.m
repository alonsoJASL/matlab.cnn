% script file : Patch creation 
% 

% clear all
% close all
% clc

fname1 = 'N2DHGOWT1_0.tif';
fname2 = 'N2DHGOWT1_1.tif';

A = imread(fname1);
[height, width] = size(A);

% we add a NaN "skirt" to the matrix.
X = -ones(height+26,width+26);
X(14:end-13,14:end-13) = A;

imindex=1:50;
Y = imread(strcat('GT_',fname1));
Y = Y>0;
Y = Y(imindex);

p1 = X(1:28,1:28);
patches = zeros(28,28,length(imindex));

filter = rand(9);

index = 1;
for i=1:height-1
    hidx = [0:27] + i;
    for j=1:width-1
        widx = [0:27] + j;
        patches(:,:,index) = X(hidx,widx);
        index = index +1;
    end
end




