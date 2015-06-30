function [J, gradJ, outsies] = softmaxRegression(theta,x,y,filtDim, numFilters, ...
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

% activation softmax
A2 = cnnSigmoid(A1,Wd,bd);
A2 = bsxfun(@minus, A2, max(A2));
A2 = exp(A2); 

probs = bsxfun(@rdivide, A2, sum(A2));
labelIndex = sub2ind(size(A2), y', 1:numImages);
oneLabels = zeros(size(A2));
oneLabels(labelIndex) = 1;

J = -sum(sum(oneLabels .* log(probs)));

if nargout > 1
    [convDim1, convDim2,~,~] = size(Fconv);
    poolDim1 = convDim1/poolSize;
    poolDim2 = convDim2/poolSize;
    
    % Go back first nn layer
    delta1 = (probs - oneLabels)/numImages;
    errorsPooled = Wd' * delta1; 
    errorsPooled = reshape(errorsPooled,[],poolSize,numFilters,numImages);
    
    % Unpool errors
    errPooling = zeros(convDim1, convDim2, numFilters, numImages);
    unpooling = ones(poolDim1, poolDim2);
    
    poolArea = poolDim1*poolDim2;
    unpooling = unpooling/poolArea;
    
    for i=1:numImages % for each image
        for j=1:numFilters % for each filter
            e = errorsPooled(:,:,j,i);
            errPooling(:,:,j,i) = kron(e,unpooling);
        end
    end
    
    % Go back on second convolutional phase
    errorsConv = errPooling .* Fconv .* (1-Fconv);
    
    % Now the gradients
    gradWd = delta1 * A1';
    gradbd = sum(delta1,2);
    
    gradWc = zeros(size(Wc));
    gradbc = zeros(size(bc));
    
    for j=1:numFilters
        e = errPooling(:,:,j,:);
        gradbc(j) = sum(e(:));
    end
    for i=1:numImages
        for j=1:numFilters
            e = errorsConv(:,:,j,i);
            errorsConv(:,:,j,i) = rot90(e,2);
        end
    end
    
    for j=1:numFilters
        gradWc_filter = zeros(size(gradWc,1), size(gradWc,2));
        for i=1:numImages
            gradWc_filter = gradWc_filter + conv2(x(:,:,i),...
                errorsConv(:,:,j,i),'valid');
        end
        gradWc(:,:,j) = gradWc_filter;
    end
        
    gradJ = [gradWc(:); gradWd(:); gradbc(:);gradbd(:)];
    
    if nargout > 2
        outsies.gradWc = gradWc;
        outsies.gradbc = gradbc;
        outsies.gradWd = gradWd;
        outsies.gradbd = gradbd;
    end
    
end

