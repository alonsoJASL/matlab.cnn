# MATLAB CNN
## Convolutional Neural Networks code.
This repository has some work on the Convolutional Neural Networks approach to
image segmentation.

The work stored in this repository depends on
[MatConvNet](https://github.com/vlfeat/matconvnet) being previously installed
 on your computer. As it worked out better for me, after you've (or forked)
 downloaded the MatConvNet project, you need to compile it. For me, the easiest
 way was by typing on a terminal:
 ```shell
make ARCH=glnxa64 MATLABROOT=/PATH/TO/MATLAB/
 ```
 or for mac users
 ```shell
make ARCH=maci64 MATLABROOT=/PATH/TO/MATLAB/
 ```
This, however, only allows you to use the CPU version of the library. In
[here](http://www.vlfeat.org/matconvnet/install-alt/) you can find a detailed
explanation of the compilation process. 
