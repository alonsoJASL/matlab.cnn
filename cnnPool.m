function [Fpool] = cnnPool(poolSize, Fconv, whichPool)
%        CONVOLUTIONAL NEURAL NETWORK POOLING
% 
% Pools the matrices stored in Fconv according to the poolDim. The accepted
% input for whichPool is the name of the function one wants to use
% (default=mean)
%

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

Fpool = zeros(poolDimension(1), poolDimension(2),numFilters, numImages);

rowsizes = poolSize*ones(poolDimension(1), 1);
colsizes = poolSize*ones(poolDimension(2), 1);

missingRows = convDim1 - sum(rowsizes);
missingCols = convDim2 - sum(colsizes);
if missingRows > 0 
    for i=1:missingRows
        rowsizes(i) = rowsizes(i) + 1;
    end
end
if missingCols > 0 
    for j=1:missingCols
        colsizes(i) = colsizes(i) + 1;
    end
end

for i=1:numFilters
    for j=1:numImages
        C = mat2cell(Fconv(:,:,i,j), rowsizes,colsizes);
        [ROWS, COLS] = size(C);
        for row=1:ROWS
            for col=1:COLS
                Fpool(row,col,i,j) = feval(whichPool,C{row,col}(:));
            end
        end
    end
end

        