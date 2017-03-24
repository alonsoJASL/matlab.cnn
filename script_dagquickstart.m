% 

%%  download a pre-trained CNN from the web (needed once)
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
  'imagenet-googlenet-dag.mat') ;
disp('Finished downloading pre-trained model.');

%% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
net.mode = 'test' ;
disp('Finished loading up the model.');


%% load and preprocess an image

% X = imread('peppers.png') ;
[X, map] = imread('corn.tif'); % peppers.png, cameraman
X = ind2rgb(X,map);

figure(1)
imagesc(X);
% Preprocess
Xpre = single(X) ; % note: 0-255 range
Xpre = imresize(Xpre, net.meta.normalization.imageSize(1:2)) ;
Xpre = bsxfun(@minus, Xpre, net.meta.normalization.averageImage) ;

%% run the CNN
net.eval({'data', Xpre}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;b
figure(1) ; clf ; imagesc(X) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;