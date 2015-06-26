function [J, gradJ] = softmaxRegression(x,y,W,b,SDG)
%        MULTINOMIAL LOGISTIC REGRESSION
% Softmax regression used for the cost computation of a neural network.
% User can choose to use Stochastic Gradient Descent, which computes the
% gradient in a different, faster way. 
%

if nargin > 4
    SDG = (SDG==true);
end

