% script file: 

%% Download a pre-trained CNN from the web (needed once).
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
  'imagenet-vgg-f.mat') ;
disp('Finished downloading pre-trained model.');

%% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;
disp('Finished tidying up the model.');

%% Obtain and preprocess an image.

[X] = imread('peppers.png');
%[X, map] = imread('corn.tif'); % peppers.png, cameraman
%X = ind2rgb(X,map);

%%
Xpre = single(X) ; % note: 255 range
Xpre = imresize(Xpre, net.meta.normalization.imageSize(1:2)) ;
Xpre = Xpre - net.meta.normalization.averageImage ;
disp('Finished Pre-processing the image X');

%% Run the CNN.
res = vl_simplenn(net, Xpre) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1); 
imagesc(X);
if size(X,3) == 1
    colormap bone;
end
title(sprintf('%s (%d), score %.3f',...
   net.meta.classes.description{best}, best, bestScore)) ;