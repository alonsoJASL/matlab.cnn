function [value, probs] = cnnTestone(theta, x,filtDim, numFilters, numClasses, ...
                                poolSize, whichPool)
%

hiddenSize = (poolSize^2)*numFilters;
numImages = size(x,3);

inWc = 1:(filtDim^2*numFilters);
inWd = (1:numClasses*hiddenSize) + filtDim^2*numFilters; 
inbc = (1:numFilters) + (numClasses*hiddenSize + filtDim^2*numFilters);
inbd = numFilters + numClasses*hiddenSize + filtDim^2*numFilters + 1;

Wc = reshape(theta(inWc), filtDim, filtDim, numFilters);
Wd = reshape(theta(inWd), numClasses, hiddenSize);
bc = theta(inbc);
bd = theta(inbd:end);

Fconv = cnnConvolve(filtDim, numFilters, x, Wc, bc);
Fpool = cnnPool(poolSize,Fconv, whichPool);

% Now the neural network layers
% Unrolling the actual input to the neural network. 
A1 = zeros(hiddenSize, numImages);
%
for i=1:numImages
    A1(:,i) = reshape(Fpool(:,:,:,i),hiddenSize, 1);
end

% activation softmax
A2 = cnnSigmoid(A1,Wd,bd);
A2 = exp(A2); 

probs = bsxfun(@rdivide, A2, sum(A2));

[~,value] = max(probs);
