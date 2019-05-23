# Progressive Growing of Variational Gradient Flow
A tensorflow implementation of VGrow by using Progressive Growing method descriped in the following paper:
* [Deep Generative Learning via Variational Gradient Flow](https://arxiv.org/abs/1901.08469).
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).

## System requirements

* We only test our model on Linux. 
* 64-bit Python 3.6 and Tensorflow 1.13.
* When you want to generate higher resolution image than 128x128, We recommend GPU with at least 16GB memory.
* NVIDIA driver 384.145  or newer, CUDA toolkit 9.0 or newer, cuDNN 7.1.2 or newer. We test the code based on the following two configuration.
  * NIVDIA driver 384.145, CUDA V9.0.176, Tesla V100
  * NVIDIA driver 410.93 , CUDA V10.0.130, RTX 2080 Ti
  
## Results
We train PGVGrow model based on different f-divergence such as KL-divergence, JS-divergence, Jef-divergence and our new porposed log-divergence. Here we only show the complete process of progressive growing based on KL-divergence. 
### KL-divergence
We train PGVGrow from low resolution (4x4) to higher resolution which depends on the training dataset. 

<table align='center'>
<tr align='center'>
<td> </td>
<td> Resolution 4x4 </td>
<td> Resolution 8x8 </td>
<td> Resolution 16x16 </td>
<td> Resolution 32x32 </td>
</tr>
<tr>
<td> Mnist </td>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
</tr>
<tr>
<td> Fashion-mnist </td>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
</tr>
<tr>
<td> Cifar-10 </td>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
<td><img src = '' height = '190px'>
</tr>
</table>

### Other-divergence

### Quantitive measure


## Usage 
### Arguments 


## Reference
The implementation is motivated based on the projects:
[1]https://github.com/tkarras/progressive_growing_of_gans
