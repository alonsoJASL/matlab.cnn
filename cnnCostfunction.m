function [J, gradJ] = cnnCostfunction(x,y,W,b,SGD)
%                       CNN COST FUNCTION
% Calculates the cost of using a set of parameters (W,b) for a specific
% training set and ground truth (x,y). User can choose which cost function
% to implement (as of this version only softmax and 'meansquareerror').
%
% 
%

if nargin < 5
   whichCost = 'softmaxRegression';
   SGD = false;
else
    try 
        aux = feval(whichCost, rand(1,3), rand(1,3), rand(3,3), rand(1,3));
        clear aux;
    catch e
        disp('Cost function not defined properly.');
        disp('Using normal softmaxRegresion');
        whichCost = 'softmaxRegresion';
    end
    if nargin > 5
        SGD = (SGD == true);
    end
end

