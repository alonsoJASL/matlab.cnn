function [J, gradJ] = cnnCostfunction(theta,x,y,filtDim, numFilters, ...
                                        numClasses, poolSize, whichPool)
%                       CNN COST FUNCTION
% Calculates the cost of using a set of parameters (W,b) for a specific
% training set and ground truth (x,y). User can choose which cost function
% to implement (as of this version only softmax and 'meansquareerror').
%
% 
%

filtDim=5;
numFilters=12;
numClasses=2;
poolSize=2;
whichPool='max';

% some parameters
epsilon = 0.01; % YOLO
lambda = 1; % YOLO as well!

