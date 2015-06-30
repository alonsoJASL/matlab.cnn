function [J, gradJ] = softmaxRegression(theta,x,y,filtDim, numFilters, ...
                                        numClasses, poolSize, ...
                                        whichPool, SDG)
%        MULTINOMIAL LOGISTIC REGRESSION
% Softmax regression used for the cost computation of a neural network.
% User can choose to use Stochastic Gradient Descent, which computes the
% gradient in a different, faster way. 
%

if nargin > 3
    SDG = (SDG==true);
else 
    SDG = false;
end

% some parameters
epsilon = 0.01; % YOLO
lambda = 1; % YOLO as well!

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
A2 = cnnSigmoid(A1,Wd,bd);

h0 = zeros(numImages,1);
for i=1:numImages
    [~,h0(i)] = max(A2(:,i));
end

% DO THE SOFTMAX REGRESSION
yA2 = zeros(size(y));
expA2 = A2;
for i=1:numImages
    yA2(i) = expA2(y(i),i);
end
sumA2 = sum(expA2,1)'; 
%
J = - sum(yA2./sumA2);

if nargout > 1
    gradJ = zeros(size(theta));
end

