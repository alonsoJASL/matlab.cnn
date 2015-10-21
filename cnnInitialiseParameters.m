function [Theta, tatt] = cnnInitialiseParameters(cnnStruct)
%           INITIALISE CNN PARAMETERS
% Create a vector of appropriate size that initialises the parameters of a
% convolutional neural network. 
% input: 
%           cnnStruct := Structure with the following parameters:
%                      -numConvL := scalar. Number of convolutional layers
%                       (i.e. CONV + POOL)
%                      -numLayers := scalar. Number of hidden fully
%                       connected layers. 
%                      -patchSize := scalar. Patches are square!
%                      -filtDim := array of dimension (numConvL x 1) that
%                       has the dimensions of the filters.
%                      -numClasses := number of out nodes.
%

finalSize = cnnStruct.

for i=1:cnnStruct.numConvL
    