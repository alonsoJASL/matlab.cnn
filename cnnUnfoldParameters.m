function [Wc, Wd, bc, bd] = cnnUnfoldParameters(theta,filtDim, numFilters, ...
                                        numClasses, poolSize)
% Unfold a value of theta into Wc, Wd, bc and bd.
%
hiddenSize = (poolSize^2)*numFilters;

inWc = 1:(filtDim^2*numFilters);
inWd = (1:numClasses*hiddenSize) + filtDim^2*numFilters; 
inbc = (1:numFilters) + (numClasses*hiddenSize + filtDim^2*numFilters);
inbd = numFilters + numClasses*hiddenSize + filtDim^2*numFilters + 1;

Wc = reshape(theta(inWc), filtDim, filtDim, numFilters);
Wd = reshape(theta(inWd), numClasses, hiddenSize);
bc = theta(inbc);
bd = theta(inbd:end);

