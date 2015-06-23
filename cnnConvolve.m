function [Fconv] = cnnConvolve(filterDim, numFilters, images, W, b)
%           CONVOLUTIONAL NEURAL NETWORK CONVOLVE
%
% Returns the convolution of the features given in "images" and using the
% W and b for the sigmoid function. 
%
% usage:  [Fconv] = cnnConvolve(filterDim, numFilters, images,W,b)
%            
% input: 
%            filterDim := Dimension of the filters to be trained. 
%           numFilters := Number of "feature maps".
%               images := Images. Arranged in a (m x n x numImages) 3D
%                         matrix.
%                    W := Actual filters, which can be stacked like before.
%                    b := bias units. One per filter, so size(b) =
%                         numfilters x 1.
%
% output:
%               Fconv := convolved features (4 dimensional matrices)
% 
% 

[imageDim1, imageDim2, numImages] = size(images);
convDim = [imageDim1 imageDim2] - filterDim + 1;

Fconv = zeros(convDim(1), convDim(2), numFilters, numImages);

% we have to check that numFilters is the same as the filters stacked on W.
if numFilters ~= size(W,3)
    numFilters = size(W,3);
    str = strcat('WARNING. numFilters is not the same as number of ',...
        'stacked filters. Using actual number of stacked filters.');
    fprintf('\n%s\nnumFilters = %d', str, numFilters);
end

for i=1:numImages
    for j=1:numFilters
        % We have to rotate the filter in order to convolve properly.
        filter = rot90(squeeze(W(:,:,j)),2);
        im = images(:,:,i);
        
        convolvedIm = conv2(im, filter, 'valid') + b(j);
        
        % we apply the sigmoid function, now to be 1/(1+exp(-Wx-b)
        Fconv(:,:,j,i) = 1./(1 + exp(-convolvedIm));
    end
end

