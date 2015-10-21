function [Patches, Annotations] = generatePatches(Im, anIm)
%

[height, width] = size(Im);

% we add a -1 "skirt" to the matrix.
X = -ones(height+26,width+26);
X(14:end-13,14:end-13) = Im;

Y = anIm>0;

Patches = zeros(28,28,height*width);

index = 1;
for i=1:height-1
    hidx = [0:27] + i;
    for j=1:width-1
        widx = [0:27] + j;
        Patches(:,:,index) = X(hidx,widx);
        index = index +1;
    end
end
Annotations = Y(:);




