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
<td><img src = 'examples/mnist4.png' height = '160px'>
<td><img src = 'examples/mnist8.png' height = '160px'>
<td><img src = 'examples/mnist16.png' height = '160px'>
<td><img src = 'examples/mnist32.png' height = '160px'>
</tr>
<tr>
<td> Fashion-mnist </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
<tr>
<td> Cifar-10 </td>
<td><img src = 'examples/cifar4.png' height = '160px'>
<td><img src = 'examples/cifar8.png' height = '160px'>
<td><img src = 'examples/cifar16.png' height = '160px'>
<td><img src = 'examples/cifar32.png' height = '160px'>
</tr>
</table>


<table align='center'>
<tr align='center'>
<td> </td>
<td> Resolution4x4 </td>
<td> Resolution8x8 </td>
<td> Resolution16x16 </td>
<td> Resolution32x32 </td>
<td> Resolution64x64 </td>
</tr>
<tr>
<td> Celeba </td>
<td><img src = 'examples/celeba4.png' height = '160px'>
<td><img src = 'examples/celeba8.png' height = '160px'>
<td><img src = 'examples/celeba16.png' height = '160px'>
<td><img src = 'examples/celeba32.png' height = '160px'>
<td><img src = 'examples/celeba64.png' height = '160px'>
</tr>
<tr>
<td> LSUN (Church) </td>
<td><img src = 'examples/church4.png' height = '160px'>
<td><img src = 'examples/church8.png' height = '160px'>
<td><img src = 'examples/church16.png' height = '160px'>
<td><img src = 'examples/church32.png' height = '160px'>
<td><img src = 'examples/church64.png' height = '160px'>
</tr>
</table>

### Other-divergence
We show all dataset final resolution results from each f-divergence.

<table align='center'>
<tr align='center'>
<td> </td>
<td> KL-divergence </td>
<td> Js-divergence </td>
<td> Jef-divergence </td>
<td> Logd-divergence </td>
</tr>
<tr>
<td> Mnist </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
<tr>
<td> Fashion-mnist </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
<tr>
<td> Cifar10 </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
<tr>
<td> Celeba </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
<tr>
<td> LSUN(Church) </td>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
<td><img src = '' height = '160px'>
</tr>
</table>

### Quantitive measure


## Usage 
### Arguments 
* `--gpu`: Specific GPU to use. *Default*: `0`
* `--dataset`: Training dataset. *Default*: `mnist`
* `--divergence`: f-divergence. *Default*: `KL`
* `--path`: Output path. *Default*: `./results`
* `--seed`: Random seed. *Default*: `1234`
* `--init_resolution`: Initial resolution of images. *Default*: `4`
* `--z_dim`: Dimension of latent vector. *Default*: `512`
* `--dur_nimg`: Number of images used for a phase. *Default*: `600000`
* `--total_nimg`: Total number of images used for training. *Default*: `12000000`
* `--pool_size`: Number of batches of a pool. *Default*: `1`
* `--T`: Number of loops for moving particles. *Default*: `1`
* `--U`: Number of loops for training D. *Default*: `1`
* `--L`: Number of loops for training G. *Default*: `1`
* `--num_row`: Number images in a line of image grid. *Default*: `10`
* `--num_line`: Number images in a row of image grid. *Default*: `10`
* `--use_gp`: Whether use gradient penalty or not. *Default*: `True`
* `--coef_gp`: Coefficient of gradient penalty. *Default*: `1`
* `--target_gp`: Target of gradient penalty. *Default*: `1`
* `--coef_smoothing`: Coefficient of generator moving average. *Default*: `0.99`
* `--resume_training`: Whether resume Training or not. *Default*: `False`
* `--resume_num`: Resume number of images. *Default*: `0`

## Reference
The implementation is motivated based on the projects:
[1]https://github.com/tkarras/progressive_growing_of_gans
