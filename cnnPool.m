function [Fpool] = cnnPool(poolSize, Fconv, whichPool)
%        CONVOLUTIONAL NEURAL NETWORK POOLING
% 
% Pools the matrices stored in Fconv according to the poolDim. The accepted
% input for whichPool is the name of the function one wants to use
% (default=mean)
%
% usage: (1) [Fpool] = cnnPool(poolSize, Fconv) -> generates pools of
%           dimension (poolSize X poolSize) for each of the images stored
%           in Fconv. This is a mean pooling. 
%        (2) [Fpool] = cnnPool(poolSize, Fconv, whichPool) -> generates the
%          'whichPool' Pooling of the images in Fconv. whichPool can be any
%           function name (string) that takes one matrix and returns a 
%           number. 
% input: 
%          poolSize := the size user wants the pool to be. So, if user
%                      wishes to have as output a lot of 2X2 matrices, 
%                       poolSize=2.
%             Fconv := 4D matrix of size:
%
%               size(Fconv) = [convDim1 convDim2 numFilters numImages]
%
%         whichPool := (optional) Function that describes matrices' inputs
%                      with one value, e.g 'mean'(default), 'std', 'max',
%                      'min' or whatever you want. 
%
% output: 
%             Fpool := 4D matrix that holds the pooling output of every
%                      convolved image.
%
%               size(Fpool) = [poolSize poolSize numFilters numImages]
% 

% Jose Alonso Solis Lemus
% 2015

if nargin < 3
    whichPool = 'mean';
else 
    try 
        aux = feval(whichPool, rand(1,10));
        clear aux;
    catch e
        disp('Undefined pooling function. Using MEAN.');
        whichPool = 'mean';
    end
end

[convDim1, convDim2, numFilters, numImages] = size(Fconv);
poolDimension = [fix(convDim1/poolSize) fix(convDim2/poolSize)];
%poolDimension = [poolSize poolSize];

Fpool = zeros(fix(convDim1/poolSize), fix(convDim2/poolSize),...
    numFilters, numImages);

he=1:poolSize:poolDimension(1)-1;
wi=1:poolSize:poolDimension(2)-1;
        
for h=1:length(he)-1
    for w=1:length(wi)-1
        for i=1:numFilters
            for j=1:numImages
                A = Fconv(he(h):he(h+1)-1,wi(w):wi(w+1)-1,i,j);
                Fpool(h,w,i,j) = feval(whichPool, A(:));
            end
        end
    end
end


% rowsizes = poolDimension(1)*ones(poolSize, 1);
% colsizes = poolDimension(2)*ones(poolSize, 1);
% missingRows = convDim1 - sum(rowsizes);
% missingCols = convDim2 - sum(colsizes);
% if missingRows > 0 
%     for i=1:missingRows
%         rowsizes(i) = rowsizes(i) + 1;
%     end
% end
% if missingCols > 0 
%     for j=1:missingCols
%         colsizes(j) = colsizes(j) + 1;
%     end
% end
% 
% for i=1:numFilters
%     for j=1:numImages
%         C = mat2cell(Fconv(:,:,i,j), rowsizes,colsizes);
%         [ROWS, COLS] = size(C);
%         for row=1:ROWS
%             for col=1:COLS
%                 Fpool(row,col,i,j) = feval(whichPool,C{row,col}(:));
%             end
%         end
%     end
% end

        