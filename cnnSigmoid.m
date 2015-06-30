function [a] = cnnSigmoid(varargin)
%       APPLY SPECIFIED SIGMOID FUNCTION
% Applies specified sigmoid function to a vector z which can be provided or
% computed through inputs: (x,W,b). User can specify any function with 
% the string whichSigmoid. User is allowed to set the function in a
% vectorized way, so in case of usage(3) and usage(4), that means that even
% if b is a vector of size(b)=[numClasses 1], here it will be properly
% computed. 
% 
% usage: (1) a = cnnSigmoid(z)
%           Uses 'regular' sigmoid function over vector z:
%                       a = 1./(1+exp(-z)). 
%           e.g. a = cnnSigmoid(rand(10,1));
%       
%        (2) a = cnnSigmoid(z, whichSigmoid)
%           Uses specific (user defined, or MATLAB available) function as
%           sigmoid function. 
%           e.g  a = feval('atan', rand(10,1));
%       
%        (3) a = cnnSigmoid(x,W,b)
%           Computes z = W*x + b before applying the specified sigmoid 
%           function.
%   
%        (4) a = cnnSigmoid(x,W,b,whichSigmoid)
%           Computes z = W*x + b with the specified 
%

switch nargin
    case 1; % z
        z = varargin{1};
        whichSigmoid = 'sigmoid';
        
    case 2; % z, whichSigmoid
        z = varargin{1};
        whichSigmoid = varargin{2};
        
    case 3; % x, W, b
        x = varargin{1};
        W = varargin{2};
        b = varargin{3};
        z = W*x + repmat(b,1,size(x,2));

        whichSigmoid = 'sigmoid';
        
    case 4; % x, W, b, whichSigmoid
        x = varargin{1};
        W = varargin{2};
        b = varargin{3};
        z = W*x + repmat(b,1,size(x,2));

        whichSigmoid = varargin{4};        
    
    otherwise
        disp('Parameters not defined properly.');
        help cnnSigmoid;
        a = [];
        return
end

try
    aux = feval(whichSigmoid, rand(1,3));
    clear aux;
catch e
    disp('Sigmoid function not specified properly, using SIGMOID:');
    disp('f(z) = 1 ./(1+exp(-z))');
    whichSigmoid = 'sigmoid';
end

a = feval(whichSigmoid, z);