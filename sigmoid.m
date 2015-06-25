function h = sigmoid(z)
% SIGMOID: computes vector h = 1./(1+exp(-z)). Operations are done
% elementwise.
%
h = 1./(1+exp(-z));